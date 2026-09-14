"""OAuth 2.1 authorisation server with PKCE and Dynamic Client Registration.

Single-user model: one shared password (set via ``MCP_AUTH_PASSWORD``) gates
every authorisation. Tokens are JWTs signed with ``JWT_SECRET``. Refresh
tokens rotate on every use.

PATCHED (restart-safe): upstream keeps registered clients and refresh tokens
in process memory, so every cold start of a sleeping free-tier host forgets
the connected client and forces a manual reconnect ~24h later. This version
makes both *stateless*:

* A registered client's ID is itself a signed JWT carrying the registration
  metadata, so ``get_client`` can reconstruct the client after a restart.
  If the SDK issued a client secret, it is replaced with one derived from
  ``JWT_SECRET`` and the client ID (never stored, never in the ID itself).
* Refresh tokens are signed JWTs (30-day expiry). Rotation is enforced with
  an in-memory revocation list, so a used refresh token is rejected for the
  life of the process; after a restart it is only bounded by its own expiry.

Authorization codes and pending logins remain in memory: they live for ten
minutes inside a single interactive flow and never need to survive a restart.
Access tokens were already stateless JWTs and are unchanged (24 hours).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

import jwt
import structlog
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

log = structlog.get_logger(__name__)

ACCESS_TOKEN_TTL_SECONDS = 24 * 60 * 60
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
AUTHORIZATION_CODE_TTL_SECONDS = 10 * 60
LOGIN_STATE_TTL_SECONDS = 10 * 60

_CLIENT_ID_PREFIX = "c1."
_REFRESH_TYP = "refresh"
_CLIENT_TYP = "client"


class InvalidLoginError(Exception):
    """Raised when /login receives the wrong password or a bad state value."""


class SimpleOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Single-user OAuth server backed by a shared password and JWT secret."""

    def __init__(self, mcp_password: str, jwt_secret: str, issuer_url: str) -> None:
        if not mcp_password:
            raise ValueError("mcp_password must be a non-empty string.")
        if not jwt_secret or len(jwt_secret) < 16:
            raise ValueError("jwt_secret must be at least 16 characters.")
        self._password = mcp_password
        self._jwt_secret = jwt_secret
        self._issuer_url = issuer_url.rstrip("/")

        self._codes: dict[str, AuthorizationCode] = {}
        # state -> (client_id, params, created_at)
        self._pending_logins: dict[str, tuple[str, AuthorizationParams, float]] = {}
        # jti -> expires_at, for refresh tokens already rotated in this process
        self._revoked_refresh: dict[str, int] = {}

    # ------------------------------------------------------------------ clients

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        if not client_id.startswith(_CLIENT_ID_PREFIX):
            return None
        try:
            payload: dict[str, Any] = jwt.decode(
                client_id[len(_CLIENT_ID_PREFIX) :],
                self._jwt_secret,
                algorithms=["HS256"],
                issuer=self._issuer_url,
                options={"require": ["iss", "iat"]},
            )
        except jwt.PyJWTError as exc:
            log.debug("oauth.client_id.invalid", error=str(exc))
            return None
        if payload.get("typ") != _CLIENT_TYP:
            return None
        meta = payload.get("c")
        if not isinstance(meta, dict):
            return None
        try:
            client = OAuthClientInformationFull(**meta)
        except Exception as exc:  # pydantic validation
            log.debug("oauth.client_id.unparseable", error=str(exc))
            return None
        client.client_id = client_id
        client.client_id_issued_at = int(payload["iat"])
        if payload.get("s"):
            client.client_secret = self._derive_client_secret(client_id)
        return client

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id is None:
            raise ValueError("Registered client must have a client_id assigned by the SDK.")
        meta = client_info.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "client_id",
                "client_secret",
                "client_id_issued_at",
                "client_secret_expires_at",
            },
        )
        now = int(time.time())
        had_secret = client_info.client_secret is not None
        encoded = jwt.encode(
            {
                "typ": _CLIENT_TYP,
                "iss": self._issuer_url,
                "iat": now,
                "c": meta,
                "s": had_secret,
            },
            self._jwt_secret,
            algorithm="HS256",
        )
        client_id = _CLIENT_ID_PREFIX + (
            encoded if isinstance(encoded, str) else encoded.decode("ascii")
        )
        # Rewrite in place: the SDK returns this same object to the client.
        client_info.client_id = client_id
        client_info.client_id_issued_at = now
        if had_secret:
            client_info.client_secret = self._derive_client_secret(client_id)
            client_info.client_secret_expires_at = None
        log.info(
            "oauth.client.registered",
            client_id_len=len(client_id),
            client_name=client_info.client_name,
            redirect_uris=[str(u) for u in (client_info.redirect_uris or [])],
            stateless=True,
        )

    def _derive_client_secret(self, client_id: str) -> str:
        return hmac.new(
            self._jwt_secret.encode("utf-8"),
            b"client-secret:" + client_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ---------------------------------------------------------------- authorize

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        client_id = client.client_id
        if client_id is None:
            raise ValueError("Client passed to authorize() has no client_id.")
        self._sweep_pending_logins()
        state = secrets.token_urlsafe(24)
        self._pending_logins[state] = (client_id, params, time.monotonic())
        return f"{self._issuer_url}/login?state={state}"

    async def complete_login(self, state: str, password: str) -> str:
        """Finish the password challenge started by ``authorize``.

        Called by the /login POST handler. Returns the redirect URL the user
        should be sent to (their client's redirect_uri with code and state).
        """
        if not secrets.compare_digest(password, self._password):
            raise InvalidLoginError("Incorrect password.")

        self._sweep_pending_logins()
        record = self._pending_logins.pop(state, None)
        if record is None:
            raise InvalidLoginError("This login link has expired. Start the connection again.")

        client_id, params, _ = record
        code_value = secrets.token_urlsafe(32)
        self._codes[code_value] = AuthorizationCode(
            code=code_value,
            scopes=params.scopes or [],
            expires_at=time.time() + AUTHORIZATION_CODE_TTL_SECONDS,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code_value,
            state=params.state,
        )

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        code = self._codes.get(authorization_code)
        if code is None:
            return None
        if code.client_id != client.client_id:
            return None
        if code.expires_at <= time.time():
            self._codes.pop(authorization_code, None)
            return None
        return code

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        # PKCE is verified by the SDK before this is called.
        client_id = self._require_client_id(client)
        self._codes.pop(authorization_code.code, None)

        access_token = self._mint_jwt(
            client_id,
            authorization_code.scopes,
            authorization_code.resource,
        )
        refresh_value = self._mint_refresh_jwt(client_id, authorization_code.scopes)
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=refresh_value,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
        )

    # ------------------------------------------------------------------ refresh

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        try:
            payload: dict[str, Any] = jwt.decode(
                refresh_token,
                self._jwt_secret,
                algorithms=["HS256"],
                audience=self._issuer_url,
                issuer=self._issuer_url,
                options={"require": ["exp", "iat", "jti", "sub"]},
            )
        except jwt.PyJWTError as exc:
            log.debug("oauth.refresh_token.invalid", error=str(exc))
            return None
        if payload.get("typ") != _REFRESH_TYP:
            return None
        if payload.get("sub") != client.client_id:
            return None
        self._sweep_revoked_refresh()
        if payload["jti"] in self._revoked_refresh:
            log.warning("oauth.refresh_token.reused", jti=payload["jti"])
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=payload["sub"],
            scopes=list(payload.get("scopes", [])),
            expires_at=int(payload["exp"]),
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Rotation: invalidate the presented refresh token for this process.
        client_id = self._require_client_id(client)
        self._revoke_refresh_jwt(refresh_token.token)

        new_scopes = scopes or refresh_token.scopes
        access_token = self._mint_jwt(client_id, new_scopes, None)
        new_refresh = self._mint_refresh_jwt(client_id, new_scopes)
        log.info("oauth.refresh.rotated")
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=new_refresh,
            scope=" ".join(new_scopes) if new_scopes else None,
        )

    # ------------------------------------------------------------------- access

    async def load_access_token(self, token: str) -> AccessToken | None:
        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=["HS256"],
                audience=self._issuer_url,
                issuer=self._issuer_url,
            )
        except jwt.PyJWTError as exc:
            log.debug("oauth.access_token.invalid", error=str(exc))
            return None
        if payload.get("typ") == _REFRESH_TYP:
            # A refresh token must never be accepted as a bearer token.
            return None

        return AccessToken(
            token=token,
            client_id=payload.get("sub", ""),
            scopes=list(payload.get("scopes", [])),
            expires_at=int(payload["exp"]) if "exp" in payload else None,
            resource=payload.get("resource"),
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, RefreshToken):
            self._revoke_refresh_jwt(token.token)
        # Access tokens are stateless JWTs. We accept the revoke call so the
        # endpoint reports success, but the token will continue to validate
        # until it expires. For a single-user server this is acceptable.

    # ------------------------------------------------------------------ helpers

    def _mint_jwt(
        self,
        client_id: str,
        scopes: list[str],
        resource: str | None,
    ) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": client_id,
            "iss": self._issuer_url,
            "aud": self._issuer_url,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            "scopes": scopes or [],
        }
        if resource is not None:
            payload["resource"] = resource
        encoded = jwt.encode(payload, self._jwt_secret, algorithm="HS256")
        return encoded if isinstance(encoded, str) else encoded.decode("ascii")

    def _mint_refresh_jwt(self, client_id: str, scopes: list[str]) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {
            "typ": _REFRESH_TYP,
            "sub": client_id,
            "iss": self._issuer_url,
            "aud": self._issuer_url,
            "iat": now,
            "exp": now + REFRESH_TOKEN_TTL_SECONDS,
            "jti": secrets.token_urlsafe(16),
            "scopes": scopes or [],
        }
        encoded = jwt.encode(payload, self._jwt_secret, algorithm="HS256")
        return encoded if isinstance(encoded, str) else encoded.decode("ascii")

    def _revoke_refresh_jwt(self, token: str) -> None:
        try:
            payload = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=["HS256"],
                audience=self._issuer_url,
                issuer=self._issuer_url,
                options={"verify_exp": False},
            )
        except jwt.PyJWTError:
            return
        jti = payload.get("jti")
        if jti:
            self._revoked_refresh[str(jti)] = int(payload.get("exp", 0))

    def _sweep_revoked_refresh(self) -> None:
        now = int(time.time())
        stale = [j for j, exp in self._revoked_refresh.items() if exp and exp <= now]
        for j in stale:
            self._revoked_refresh.pop(j, None)

    @staticmethod
    def _require_client_id(client: OAuthClientInformationFull) -> str:
        if client.client_id is None:
            raise ValueError("Client has no client_id.")
        return client.client_id

    def _sweep_pending_logins(self) -> None:
        cutoff = time.monotonic() - LOGIN_STATE_TTL_SECONDS
        stale = [s for s, (_, _, ts) in self._pending_logins.items() if ts < cutoff]
        for s in stale:
            self._pending_logins.pop(s, None)
