"""C4 REST API. `router` (file-based store, built on first access) or `build_router(...)`; see router.py."""
from .router import build_router

__all__ = ['build_router', 'router']
_router = None


def __getattr__(name):
    global _router
    if name == 'router':
        if _router is None:
            _router = build_router()
        return _router
    raise AttributeError(name)
