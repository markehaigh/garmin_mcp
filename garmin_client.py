"""Wrapper around the unofficial ``garminconnect`` Python package.

Responsibilities:
    * Lazy login on the first tool call.
    * Persist Garmin OAuth tokens so we do not log in on every cold start.
    * Re-authenticate transparently when the saved session has expired.
    * Rate limit to one outbound call every ``rate_limit_seconds`` seconds
      so we do not look like a misbehaving client.
    * Retry network blips with exponential backoff.

PATCHED (restart-safe token persistence): Garmin rotates its refresh token
and the original stops working after a few days. On a host whose disk is
wiped on every restart, re-seeding the token file from a static environment
variable therefore fails once the original refresh token dies. This version
keeps the *latest* token bundle in a key-value store (``REDIS_URL``) and
writes it back whenever the underlying client rotates it:

    load order:  key-value store  ->  token file on disk (seeded from env)
    save:        after every successful login and after any call that
                 changed the token bundle

If the key-value store is unreachable the wrapper degrades to the file-only
behaviour and logs a warning; it never blocks a Garmin call.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

TOKEN_STORE_KEY = os.getenv("GARMIN_TOKEN_STORE_KEY", "garmin-mcp:tokens:v1")


class GarminClientError(RuntimeError):
    """Raised when a Garmin call fails after retries."""


class GarminAuthError(GarminClientError):
    """Raised when re-authentication is required but cannot succeed."""


_RETRYABLE_NETWORK_EXC = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

# The wrapper dispatches by method name via getattr, so it can in principle
# invoke ANY method on garminconnect.Garmin (including writes/deletes). To keep
# the server read-only by *enforcement* rather than by call-site convention,
# only methods on these allowlists may be called. Write methods additionally
# require ``allow_writes=True`` (driven by GARMIN_WRITE_ENABLED).
_READ_METHODS: frozenset[str] = frozenset(
    {
        "get_sleep_data",
        "get_activities",
        "get_activity",
        "get_activity_splits",
        "get_activity_hr_in_timezones",
        "get_training_status",
        "get_hrv_data",
        "get_body_battery",
        "get_user_summary",
        "get_rhr_day",
        "get_stress_data",
        "get_training_readiness",
        "get_max_metrics",
        "get_race_predictions",
        "get_personal_record",
        "get_body_composition",
        "get_respiration_data",
        "get_weekly_intensity_minutes",
        "get_weekly_steps",
        "get_weekly_stress",
        "get_workouts",
        "get_workout_by_id",
        "get_activity_exercise_sets",
        "get_endurance_score",
        "get_hill_score",
        "get_activity_weather",
    }
)
# Deliberately minimal: create, delete, and calendar schedule/unschedule of
# library workouts only. No activity or profile writes.
_WRITE_METHODS: frozenset[str] = frozenset(
    {"upload_workout", "delete_workout", "schedule_workout", "unschedule_workout"}
)


class _TokenStore:
    """Tiny key-value persistence for the Garmin token bundle.

    Backed by Redis-compatible ``REDIS_URL`` when set; otherwise inert.
    Every operation is best-effort and swallows its own errors.
    """

    def __init__(self, url: str | None, key: str) -> None:
        self._url = url
        self._key = key
        self._client: Any | None = None
        self._disabled = not url

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _conn(self) -> Any | None:
        if self._disabled:
            return None
        if self._client is None:
            try:
                import redis  # type: ignore[import-not-found]

                self._client = redis.Redis.from_url(
                    self._url,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    decode_responses=True,
                )
            except Exception as exc:
                log.warning("garmin.tokenstore.unavailable", error=str(exc)[:200])
                self._disabled = True
                return None
        return self._client

    def load(self) -> str | None:
        conn = self._conn()
        if conn is None:
            return None
        try:
            value = conn.get(self._key)
        except Exception as exc:
            log.warning("garmin.tokenstore.load_failed", error=str(exc)[:200])
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value or None

    def save(self, blob: str) -> bool:
        conn = self._conn()
        if conn is None:
            return False
        try:
            conn.set(self._key, blob)
            return True
        except Exception as exc:
            log.warning("garmin.tokenstore.save_failed", error=str(exc)[:200])
            return False


class GarminClient:
    """Async-friendly facade over the synchronous ``garminconnect`` library."""

    def __init__(
        self,
        email: str,
        password: str,
        token_dir: str = "/tmp/garth",
        rate_limit_seconds: float = 2.0,
        allow_writes: bool = False,
    ) -> None:
        # Empty creds are allowed: the server can run on saved tokens alone
        # (seeded by `garmin-mcp login`). They are only required when the saved
        # session expires and a fresh login is needed.
        self._email = email
        self._password = password
        self._token_dir = Path(token_dir)
        self._rate_limit_seconds = rate_limit_seconds
        self._allow_writes = allow_writes
        self._client: Any | None = None
        self._login_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._last_call_at = 0.0
        self._store = _TokenStore(os.getenv("REDIS_URL"), TOKEN_STORE_KEY)
        # Store-seeded bundle lives beside, not over, the env-seeded file so a
        # dead store bundle can never mask a good freshly-seeded one.
        self._store_dir = self._token_dir / "store"
        self._last_saved_blob: str | None = None

    # ------------------------------------------------------------ persistence

    def _current_blob(self) -> str | None:
        if self._client is None:
            return None
        try:
            return str(self._client.client.dumps())
        except Exception:
            return None

    def _persist_if_changed(self, reason: str) -> None:
        blob = self._current_blob()
        if not blob or blob == self._last_saved_blob:
            return
        # Keep the on-disk copy current too, so the same process can resume
        # without the store.
        try:
            self._client.client.dump(str(self._token_dir))  # type: ignore[union-attr]
        except Exception:
            pass
        if self._store.save(blob):
            log.info("garmin.tokens.persisted", reason=reason)
        self._last_saved_blob = blob

    def _seed_file_from_store(self) -> bool:
        """Write the store's token bundle into the store dir. True if done."""
        blob = self._store.load()
        if not blob:
            return False
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._store_dir, 0o700)
            except OSError:
                pass
            target = self._store_dir / "garmin_tokens.json"
            tmp = target.with_suffix(".json.tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp, target)
            self._last_saved_blob = blob
            log.info("garmin.tokens.seeded_from_store")
            return True
        except Exception as exc:
            log.warning("garmin.tokenstore.seed_failed", error=str(exc)[:200])
            return False

    # ------------------------------------------------------------------ login

    async def _try_resume(self, Garmin: Any, source: str, token_dir: Path) -> Any | None:
        client = Garmin(email=self._email, password=self._password, return_on_mfa=False)
        try:
            await asyncio.to_thread(client.login, str(token_dir))
        except Exception as exc:  # broad, garminconnect raises a mix of types
            log.info(
                "garmin.login.resume_failed",
                source=source,
                reason=type(exc).__name__,
                detail=str(exc)[:200],
            )
            return None
        log.info("garmin.login.resumed", source=source, token_dir=str(token_dir))
        return client

    async def _ensure_logged_in(self, force: bool = False) -> Any:
        async with self._login_lock:
            if self._client is not None and not force:
                return self._client

            from garminconnect import Garmin  # local import keeps tests light

            self._token_dir.mkdir(parents=True, exist_ok=True)

            client: Any | None = None

            # 1. Newest bundle from the key-value store (survives host restarts).
            if self._store.enabled and await asyncio.to_thread(self._seed_file_from_store):
                client = await self._try_resume(Garmin, "store", self._store_dir)

            # 2. Whatever is on disk (seeded from the environment at boot).
            if client is None:
                self._last_saved_blob = None
                client = await self._try_resume(Garmin, "file", self._token_dir)

            # 3. Fresh credential login, if credentials were provided.
            if client is None:
                if not self._email or not self._password:
                    raise GarminAuthError(
                        "Saved Garmin session is invalid and no credentials are set. "
                        "Generate a fresh token (garmin-mcp login, or the notebook "
                        "cell in the runbook) and set GARMIN_TOKENS_JSON again."
                    )
                client = Garmin(
                    email=self._email, password=self._password, return_on_mfa=False
                )
                try:
                    await asyncio.to_thread(client.login)
                except Exception as login_exc:
                    raise GarminAuthError(f"Garmin login failed: {login_exc}") from login_exc
                log.info("garmin.login.success")

            self._client = client
            self._persist_if_changed("login")
            return client

    # ------------------------------------------------------------------- calls

    async def _rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_at
            if elapsed < self._rate_limit_seconds:
                await asyncio.sleep(self._rate_limit_seconds - elapsed)
            self._last_call_at = time.monotonic()

    @staticmethod
    def _looks_like_auth_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("401", "403", "unauthorized", "forbidden", "expired", "auth", "login")
        )

    async def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the underlying Garmin client.

        We retry on transient network errors. On what looks like an auth
        error we drop the cached client, re-login, and try one more time.
        """
        if method_name not in _READ_METHODS and method_name not in _WRITE_METHODS:
            raise GarminClientError(
                f"Method '{method_name}' is not on the allowlist; refusing to call it."
            )
        if method_name in _WRITE_METHODS and not self._allow_writes:
            raise GarminAuthError(
                f"Write method '{method_name}' blocked: writes are disabled "
                "(set GARMIN_WRITE_ENABLED=1 to enable)."
            )
        # Writes are non-idempotent POSTs: a network timeout AFTER the request
        # commits server-side, then a blind retry, would create a duplicate. So
        # do not retry writes on transient network errors. (The auth-error retry
        # below is still safe for writes — a 401/403 means the call was rejected
        # before committing, so re-login + one retry cannot duplicate.)
        is_write = method_name in _WRITE_METHODS
        max_attempts = 1 if is_write else 3

        async def _invoke() -> Any:
            await self._rate_limit()
            client = await self._ensure_logged_in()
            method = getattr(client, method_name, None)
            if method is None:
                raise GarminClientError(
                    f"garminconnect.Garmin has no method named '{method_name}'."
                )
            result = await asyncio.to_thread(method, *args, **kwargs)
            # The library refreshes its own tokens before requests; capture
            # any rotation so a later restart resumes from the newest bundle.
            self._persist_if_changed("call")
            return result

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                retry=retry_if_exception_type(_RETRYABLE_NETWORK_EXC),
                reraise=True,
            ):
                with attempt:
                    return await _invoke()
        except RetryError as retry_exc:
            raise GarminClientError(
                f"Garmin call '{method_name}' failed after retries: {retry_exc}"
            ) from retry_exc
        except _RETRYABLE_NETWORK_EXC as net_exc:
            raise GarminClientError(f"Garmin call '{method_name}' failed: {net_exc}") from net_exc
        except GarminAuthError:
            raise
        except Exception as exc:
            if self._looks_like_auth_error(exc):
                log.warning("garmin.auth.expired", method=method_name, error=str(exc)[:200])
                self._client = None
                try:
                    return await _invoke()
                except Exception as retry_exc:
                    raise GarminAuthError(
                        f"Garmin re-auth attempt failed for '{method_name}': {retry_exc}"
                    ) from retry_exc
            raise GarminClientError(
                f"Garmin call '{method_name}' raised {type(exc).__name__}: {exc}"
            ) from exc

        # Unreachable; AsyncRetrying either returns or raises.
        raise GarminClientError(
            f"Unreachable: '{method_name}' produced no result."
        )  # pragma: no cover
