"""Registers every torch -> Equinox module converter.

Import requires torch (conversion only); inference code never imports this
module."""

from __future__ import annotations

from . import backend
from .nn import features, layers


def register_all() -> None:
    from .nn import composites, transformer

    backend.register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()


register_all()
