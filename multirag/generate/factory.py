"""Backend lookup by name.

Adding a provider means writing one class and adding one line here. Nothing
else in MultiRAG needs to know it exists.
"""

from multirag import config
from multirag.generate.base import Generator


class UnknownBackendError(ValueError):
    """Raised when the configured backend name is not registered."""


def load_generator(name: str | None = None, **kwargs) -> Generator:
    """Build the named backend, defaulting to config.GENERATOR_BACKEND.

    Imports happen inside each branch so an unused backend's dependencies are
    never required - `echo` works with nothing installed beyond Pillow.
    """
    name = (name or config.GENERATOR_BACKEND).lower()

    if name == "echo":
        from multirag.generate.echo import EchoGenerator

        return EchoGenerator()

    if name == "claude":
        from multirag.generate.claude import ClaudeGenerator

        return ClaudeGenerator(**kwargs)

    raise UnknownBackendError(
        f"unknown generator backend {name!r}; known: 'claude', 'echo'. "
        "Set MULTIRAG_GENERATOR, or register your own in generate/factory.py"
    )
