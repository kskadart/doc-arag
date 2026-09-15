"""
Who is calling the API and what they may do.

Authentication itself happens at the edge: Caddy asks Authelia about every
request (forward_auth) and, when the session is valid, forwards the request
with Remote-User / Remote-Groups / Remote-Email / Remote-Name headers set from
Authelia's answer. This module turns those headers into a CurrentUser and
enforces the group-based authorization the edge rules only approximate.

Trust boundary. The headers alone prove nothing: the API sits on a docker
network shared with other containers (the Next.js client, sterility-ai,
rag-services), any of which could send `Remote-Groups: admins` straight to
api:8103. Caddy therefore also injects X-Auth-Proxy-Secret, a value only the
edge and this service know (AUTH_PROXY_SECRET), and identity headers are
honoured only on requests carrying it. Binding the host port to 127.0.0.1
merely keeps the LAN out.

With AUTH_TRUSTED_HEADERS=false (default: local runs, tests) every request is
an anonymous administrator; the API logs a warning at startup in that mode.
"""

import hmac
import logging
from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException, status

from src.docarag.settings import settings

logger = logging.getLogger(__name__)

AUTH_MODE_NONE = "none"
AUTH_MODE_TRUSTED_HEADERS = "trusted-headers"


def auth_mode() -> str:
    """Name reported by GET /me so the UI can tell 'login disabled' from 'admin'."""
    return (
        AUTH_MODE_TRUSTED_HEADERS if settings.auth_trusted_headers else AUTH_MODE_NONE
    )


@dataclass(frozen=True)
class CurrentUser:
    """Identity asserted by the edge proxy for the current request.

    A plain value object: `is_admin` is decided once, when the request is
    resolved, so the instance does not depend on process settings afterwards.
    """

    username: str
    display_name: str | None = None
    email: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False


# Login disabled: nobody is identified, everything is allowed, no group is faked
ANONYMOUS_ADMIN = CurrentUser(username="anonymous", is_admin=True)


def _header_text(value: str | None) -> str | None:
    """Header value as text.

    Starlette decodes header bytes as latin-1 while Authelia sends UTF-8, so a
    display name like 'Кирилл' arrives mojibaked; the round trip restores it.
    """
    if value is None:
        return None
    try:
        value = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    value = value.strip()
    return value or None


def _parse_groups(header: str | None) -> frozenset[str]:
    """Authelia sends groups comma-separated; tolerate spaces and empties."""
    text = _header_text(header)
    if not text:
        return frozenset()
    return frozenset(g.strip() for g in text.split(",") if g.strip())


def _not_authenticated() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )


def get_current_user(
    remote_user: str | None = Header(default=None, alias="Remote-User"),
    remote_groups: str | None = Header(default=None, alias="Remote-Groups"),
    remote_email: str | None = Header(default=None, alias="Remote-Email"),
    remote_name: str | None = Header(default=None, alias="Remote-Name"),
    proxy_secret: str | None = Header(default=None, alias="X-Auth-Proxy-Secret"),
) -> CurrentUser:
    """Resolve the caller; 401 unless the edge proxy vouched for somebody."""
    if not settings.auth_trusted_headers:
        return ANONYMOUS_ADMIN

    expected = settings.auth_proxy_secret.get_secret_value()
    if not proxy_secret or not hmac.compare_digest(proxy_secret, expected):
        # Either a caller bypassing Caddy or an edge deployed without the secret
        logger.warning("request without a valid X-Auth-Proxy-Secret rejected")
        raise _not_authenticated()

    username = _header_text(remote_user)
    if not username:
        raise _not_authenticated()

    groups = _parse_groups(remote_groups)
    return CurrentUser(
        username=username,
        display_name=_header_text(remote_name),
        email=_header_text(remote_email),
        groups=groups,
        is_admin=settings.auth_admin_group in groups,
    )


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Document management is limited to the administrator group."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Membership in group '{settings.auth_admin_group}' required",
        )
    return user
