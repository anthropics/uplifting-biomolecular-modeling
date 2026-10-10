"""Ownership of geometry workspaces and their weight-pack finalizers."""
import contextlib
import threading
import weakref


class PayloadCache(dict):
    """A shared geometry cache whose weight finalizers cannot keep it alive."""
    __slots__ = ("__weakref__", "_pack_finalizers")

    def __init__(self):
        super().__init__()
        self._pack_finalizers = {}


    def detach_owner(self, key):
        registration = self._pack_finalizers.pop(key, None)
        if registration is not None:
            registration['owner'].detach()
            registration['cache'].detach()

    def bind_owner(self, key, owner, w_ptr, evict):
        self.detach_owner(key)
        registration = {}
        finalizer = weakref.finalize(owner, _evict_weak_cache, weakref.ref(self), key, w_ptr, evict, registration)
        # Whichever object dies first removes the other registration.
        registration['cache'] = weakref.finalize(self, finalizer.detach)
        registration['owner'] = finalizer
        self._pack_finalizers[key] = registration


def _evict_weak_cache(cache_ref, key, w_ptr, evict, registration):
    registration['cache'].detach()
    cache = cache_ref()
    if cache is not None:
        evict(cache, key, w_ptr, 'gc')


_CAPTURE_CACHES = threading.local()


@contextlib.contextmanager
def retain_payload_caches():
    """Retain every geometry touched in this scope, including mid-capture LRU evictions.

    The caller takes ownership of the yielded mapping when capture succeeds.
    Nested scopes retain complete sets; unrelated threads are isolated.
    """
    scopes = getattr(_CAPTURE_CACHES, "scopes", None)
    if scopes is None:
        scopes = _CAPTURE_CACHES.scopes = []
    caches = {}
    scopes.append(caches)
    try:
        yield caches
    finally:
        scopes.pop()


