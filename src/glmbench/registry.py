"""Generic name->factory registries — composition over inheritance.

Every pluggable component (adapter, task, runner-backend, metric, leaderboard
formatter) registers itself here under a *kind* and a *name* so the runner can
resolve names to factories without hard-coded imports. Adding a new adapter or
task is a localized change: drop one module, register it, done — no edits to the
runner.

Usage:
    from glmbench import registry

    @registry.register("task", "rnagym-dms")
    class RNAGymDMSTask: ...

    cls = registry.resolve("task", "rnagym-dms")   # returns the factory
    obj = registry.build("task", "rnagym-dms")     # instantiate with kwargs
    names = registry.list("task")                  # ["rnagym-dms", ...]

The kind axis keeps namespaces separate: an adapter and a task may share a name.
Registration is fail-loud — re-registering an existing (kind, name) raises.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

# kind -> { name -> factory }
_registries: dict[str, dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable[[_T], _T]:
    """Decorator registering a factory under (*kind*, *name*).

    The factory is typically a class but may be any callable. Raises ValueError
    if the (kind, name) pair is already taken — fail loud, not silent.
    """

    def decorator(factory: _T) -> _T:
        reg = _registries.setdefault(kind, {})
        if name in reg:
            raise ValueError(
                f"Registry name '{name}' is already taken for kind '{kind}' "
                f"by {reg[name]!r}. Pick a unique name or unregister it first."
            )
        reg[name] = factory
        return factory

    return decorator


def resolve(kind: str, name: str) -> Any:
    """Return the factory registered under (*kind*, *name*).

    Raises KeyError with a helpful message if the kind or name is unknown.
    """
    reg = _registries.get(kind, {})
    if name not in reg:
        available = sorted(reg.keys())
        raise KeyError(
            f"'{name}' not found in registry kind '{kind}'. "
            f"Available names: {available if available else '(empty)'}"
        )
    return reg[name]


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Resolve (*kind*, *name*) and instantiate it with *kwargs*."""
    return resolve(kind, name)(**kwargs)


def list(kind: str) -> builtins.list[str]:  # noqa: A001 - intentional public name
    """Return a sorted list of names registered under *kind* (empty if none)."""
    return sorted(_registries.get(kind, {}).keys())


def kinds() -> builtins.list[str]:
    """Return a sorted list of all registered kinds."""
    return sorted(_registries.keys())


def unregister(kind: str, name: str) -> None:
    """Remove a (kind, name) from the registry. Useful in tests."""
    _registries.get(kind, {}).pop(name, None)


def clear(kind: str | None = None) -> None:
    """Clear one kind, or the entire registry when *kind* is None. Tests only."""
    if kind is None:
        _registries.clear()
    else:
        _registries.pop(kind, None)
