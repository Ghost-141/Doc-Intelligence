from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import Settings, get_settings


def build_limiter(settings: Settings) -> Limiter:
    return Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


limiter = build_limiter(get_settings())
