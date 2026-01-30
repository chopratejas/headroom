"""Main CLI entry point for Headroom."""

import click


def get_version() -> str:
    """Get the current version."""
    try:
        from headroom import __version__

        return __version__
    except ImportError:
        return "unknown"


@click.group()
@click.version_option(version=get_version(), prog_name="headroom")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Headroom - The Context Optimization Layer for LLM Applications.

    Manage memories, run the optimization proxy, and analyze metrics.

    \b
    Examples:
        headroom proxy              Start the optimization proxy
        headroom memory list        List stored memories
        headroom memory stats       Show memory statistics
    """
    ctx.ensure_object(dict)


# Import subcommands - these register themselves with the main group
def _register_commands() -> None:
    """Register all subcommand groups."""
    from . import (
        evals,  # noqa: F401
        memory,  # noqa: F401
        proxy,  # noqa: F401
    )


_register_commands()

if __name__ == "__main__":
    main()
