"""stand-in jax._src.compilation_cache (tests only): the two names lever compilecache inspects and its reset."""
_cache_initialized = False
_cache = None


def reset_cache():
    global _cache_initialized, _cache
    _cache_initialized = False
    _cache = None
