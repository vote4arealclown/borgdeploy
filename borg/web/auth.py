"""Session-based password auth for the Borg dashboard."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from fastapi import Request
from itsdangerous import BadSignature, TimestampSigner
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import RedirectResponse, Response

from borg.config import settings


class AuthManager:
    """Simple password gate using signed session cookies."""

    COOKIE_NAME = "borg_session"

    def __init__(self) -> None:
        self._secret = settings.borg_password or "borg-dev-secret-change-me"
        self._signer = TimestampSigner(self._secret)
        self._session_ttl = timedelta(hours=24)

    def is_enabled(self) -> bool:
        return bool(settings.borg_password)

    def login(self, password: str) -> Optional[str]:
        if password == settings.borg_password:
            token = self._signer.sign("authenticated").decode("utf-8")
            return token
        return None

    def verify(self, request: Request) -> bool:
        if not self.is_enabled():
            return True
        token = request.cookies.get(self.COOKIE_NAME)
        if not token:
            return False
        try:
            self._signer.unsign(token, max_age=int(self._session_ttl.total_seconds()))
            return True
        except BadSignature:
            return False

    def logout(self, response: Response) -> None:
        response.delete_cookie(self.COOKIE_NAME)


auth = AuthManager()


class AuthMiddleware(BaseHTTPMiddleware):
    """Protect HTML routes and API endpoints (except login and static files)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        # Public routes
        if path in ("/login", "/logout") or path.startswith("/static/"):
            return await call_next(request)
        # Health endpoints can stay open
        if path in ("/healthz",):
            return await call_next(request)
        if auth.is_enabled() and not auth.verify(request):
            if request.headers.get("accept", "").startswith("application/json") or path.startswith("/api/"):
                return Response("Unauthorized", status_code=401)
            return RedirectResponse(url="/login", status_code=302)
        return await call_next(request)
