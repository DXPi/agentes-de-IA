import click
from main import run_cli_demo

from app.client_lib.utils import coroutine, handle_exceptions


@click.group()
def autogpt():
    """Temporary command group for v2 commands."""
    pass


@autogpt.command()
@click.option(
    "--pdb",
    is_flag=True,
    help="Drop into a debugger if an error is raised.",
)
@coroutine
async def run(pdb: bool) -> None:
    """Run the AFAAS-Demo agent."""
    click.echo("Running AFAAS-Demo agent...")
    main = handle_exceptions(run_cli_demo, with_debugger=pdb)
    await main()


if __name__ == "__main__":
    autogpt()