"""
Who is calling the API and what they may do.

Authentication itself happens at the edge: Caddy asks Authelia about every
request (forward_auth) and, when the session is valid, forwards the request
with Remote-User / Remote-Groups / Remote-Email / Remote-Name headers set from
Authelia's answer. This module turns those headers into a CurrentUser and
enforces the group-based authorization the edge rules only approximate.

The headers are trusted only because the API is not reachable except through
that proxy (compose binds the port to 127.0.0.1) and Caddy replaces any
client-supplied value with Authelia's. With AUTH_MODE=none (local runs, tests)
every request is an anonymous administrator.
"""

from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException, status

from src.docarag.settings import settings

ANONYMOUS_USERNAME = "anonymous"


@dataclass(frozen=True)
class CurrentUser:
    """Identity asserted by the edge proxy for the current request."""

    username: str
    display_name: str | None = None
    email: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_admin(self) -> bool:
        return settings.auth_admin_group in self.groups


def _parse_groups(header: str | None) -> frozenset[str]:
    """Authelia sends groups comma-separated; tolerate spaces and empties."""
    if not header:
        return frozenset()
    return frozenset(g.strip() for g in header.split(",") if g.strip())


def get_current_user(
    remote_user: str | None = Header(default=None, alias="Remote-User"),
    remote_groups: str | None = Header(default=None, alias="Remote-Groups"),
    remote_email: str | None = Header(default=None, alias="Remote-Email"),
    remote_name: str | None = Header(default=None, alias="Remote-Name"),
) -> CurrentUser:
    """Resolve the caller; 401 when the proxy asserted nobody."""
    if settings.auth_mode == "none":
        return CurrentUser(
            username=ANONYMOUS_USERNAME,
            groups=frozenset({settings.auth_admin_group}),
        )

    if not remote_user or not remote_user.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    return CurrentUser(
        username=remote_user.strip(),
        display_name=remote_name.strip() if remote_name else None,
        email=remote_email.strip() if remote_email else None,
        groups=_parse_groups(remote_groups),
    )


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Document management is limited to the administrator group."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Membership in group '{settings.auth_admin_group}' required",
        )
    return user
