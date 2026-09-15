"""
Identity headers for scripts that call the API directly, bypassing Caddy.

On a deployment with AUTH_TRUSTED_HEADERS=true the API accepts a request only
when it carries the proxy secret and a Remote-User; the scripts act as a
service account in the admin group. The secret is read from the environment
or, failing that, from the .env in the current directory (the same file the
api container uses). Without a secret no headers are sent, which is right for
a local stack where login is disabled.
"""

import os
from pathlib import Path

SECRET_VAR = "AUTH_PROXY_SECRET"
DEFAULT_SCRIPT_USER = "scripts"


def _from_dotenv(key: str, path: Path = Path(".env")) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("'\"") or None
    return None


def api_headers() -> dict[str, str]:
    """Headers that make the API treat the script as an administrator."""
    secret = os.environ.get(SECRET_VAR) or _from_dotenv(SECRET_VAR)
    if not secret:
        return {}
    return {
        "X-Auth-Proxy-Secret": secret,
        "Remote-User": os.environ.get("AUTH_SCRIPT_USER", DEFAULT_SCRIPT_USER),
        "Remote-Groups": os.environ.get("AUTH_ADMIN_GROUP", "admins"),
    }
