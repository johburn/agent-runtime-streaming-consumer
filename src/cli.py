"""Interactive CLI and stream viewer for Google Cloud Agent Runtime SSE streaming."""

import argparse
import asyncio
import sys
import time
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .auth import GoogleAuthTokenProvider
from .client import AgentRuntimeClient
from .config import AgentRuntimeConfig
from .formatting import render_event
from .models import (
    ClientMode,
    DoneEvent,
    ErrorEvent,
    PayloadFormat,
    RunConfig,
    StreamingMode,
    TextDelta,
)

console = Console()


def print_banner(config: AgentRuntimeConfig) -> None:
    """Displays startup banner with session and connection parameters."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="white")

    mode_label = "Vertex AI SDK" if config.client_mode == ClientMode.VERTEX_SDK else "Direct REST streamQuery"
    table.add_row("Client Mode:", f"[bold magenta]{mode_label}[/bold magenta] ({config.client_mode.value})")
    table.add_row("Project ID:", config.project_id or "[yellow]Auto-detect / Not set[/yellow]")
    table.add_row("Location:", config.location)
    table.add_row("Reasoning Engine:", config.reasoning_engine_id or "[red]Not configured[/red]")
    table.add_row("Streaming Mode:", f"[bold green]{config.streaming_mode.value.upper()}[/bold green] (Server-Sent Events)")
    table.add_row("Payload Format:", config.payload_format.value)
    table.add_row("User ID:", config.user_id)
    table.add_row("Session ID:", f"[dim]{config.ensure_session_id()}[/dim]")

    panel = Panel(
        table,
        title="🚀 [bold green]Agent Runtime SSE Streaming Client[/bold green]",
        subtitle="Commands: [bold]/new[/bold] (new session) | [bold]/session[/bold] | [bold]/raw[/bold] | [bold]/exit[/bold]",
        border_style="blue",
        expand=False,
    )
    console.print(panel)


async def execute_stream_turn(
    client: AgentRuntimeClient,
    message: str,
    show_raw: bool = False,
) -> None:
    """Runs a single query turn and streams events to console."""
    start_time = time.time()
    console.print("\n[bold blue]🤖 Agent:[/bold blue] ", end="")

    token_count = 0
    async for event in client.stream_query(message=message):
        if show_raw:
            console.print(f"\n[dim yellow]RAW SSE:[/dim yellow] {event.raw_data}")

        render_event(event, console)

        if isinstance(event, TextDelta):
            token_count += 1
        elif isinstance(event, ErrorEvent):
            console.print()
            return
        elif isinstance(event, DoneEvent):
            pass

    elapsed = time.time() - start_time
    console.print(f"\n[dim]({elapsed:.2f}s)[/dim]\n")


async def run_interactive_session(
    client: AgentRuntimeClient,
    initial_prompt: Optional[str] = None,
    show_raw: bool = False,
) -> None:
    """Runs the interactive REPL loop."""
    print_banner(client.config)

    if initial_prompt:
        console.print(f"[bold green]User:[/bold green] {initial_prompt}")
        await execute_stream_turn(client, initial_prompt, show_raw=show_raw)

    while True:
        try:
            user_input = console.input("[bold green]User:[/bold green] ").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[bold blue]Goodbye![/bold blue]")
                break
            elif user_input.lower() in ("/new", "/reset"):
                new_id = client.new_session()
                console.print(f"\n🔄 [bold green]New session started:[/bold green] [dim]{new_id}[/dim]\n")
                continue
            elif user_input.lower() == "/session":
                print_banner(client.config)
                continue
            elif user_input.lower() == "/raw":
                show_raw = not show_raw
                state_str = "[green]ENABLED[/green]" if show_raw else "[yellow]DISABLED[/yellow]"
                console.print(f"\nRaw SSE payload view: {state_str}\n")
                continue
            elif user_input.lower() == "/help":
                console.print("\n[bold]Available Commands:[/bold]")
                console.print("  [cyan]/new[/cyan] or [cyan]/reset[/cyan] - Start a fresh conversation session with a new session ID")
                console.print("  [cyan]/session[/cyan]      - Display current connection and session settings")
                console.print("  [cyan]/raw[/cyan]          - Toggle display of raw JSON SSE chunks")
                console.print("  [cyan]/exit[/cyan]         - Quit application\n")
                continue

            await execute_stream_turn(client, user_input, show_raw=show_raw)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold blue]Session terminated. Goodbye![/bold blue]")
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Client application for Google Cloud Agent Runtime / Agent Engine using SSE streaming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional single-shot prompt to send. If not specified, launches interactive mode.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch in interactive multi-turn chat mode",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["sdk", "api"],
        default=None,
        help="Client invocation backend: 'sdk' (Vertex AI SDK) or 'api' (Direct REST :streamQuery URL)",
    )
    parser.add_argument("-p", "--project", help="Google Cloud Project ID")
    parser.add_argument("-l", "--location", default=None, help="Google Cloud Location / Region (e.g. us-central1)")
    parser.add_argument("-e", "--engine-id", help="Reasoning Engine / Agent Runtime ID or full resource name")
    parser.add_argument("-s", "--session-id", help="Custom Session ID")
    parser.add_argument("-u", "--user-id", help="Custom User ID")
    parser.add_argument(
        "--streaming-mode",
        choices=["sse", "none", "bidi"],
        default="sse",
        help="Streaming mode (default: sse)",
    )
    parser.add_argument(
        "--format",
        choices=["standard", "adk_parts", "raw"],
        default="standard",
        help="Request payload format for Agent Runtime",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw SSE JSON events")
    parser.add_argument("--creds", help="Path to Service Account JSON key file")
    parser.add_argument("--token", help="Explicit OAuth2 Bearer token override")
    parser.add_argument("--endpoint", help="Custom API endpoint override")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = AgentRuntimeConfig.from_deployment_metadata()

    if args.mode:
        config.client_mode = ClientMode(args.mode)
    if args.project:
        config.project_id = args.project
    if args.location:
        config.location = args.location
    if args.engine_id:
        config.reasoning_engine_id = args.engine_id
    if args.session_id:
        config.session_id = args.session_id
    if args.user_id:
        config.user_id = args.user_id
    if args.streaming_mode:
        config.streaming_mode = StreamingMode(args.streaming_mode)
        config.run_config = RunConfig(streaming_mode=config.streaming_mode)
    if args.format:
        config.payload_format = PayloadFormat(args.format)
    if args.creds:
        config.credentials_path = args.creds
    if args.token:
        config.access_token = args.token
    if args.endpoint:
        config.api_endpoint_override = args.endpoint

    # Build client
    auth_provider = GoogleAuthTokenProvider(
        credentials_path=config.credentials_path,
        access_token_override=config.access_token,
    )
    client = AgentRuntimeClient(config=config, auth_provider=auth_provider)

    # If single-shot prompt provided without -i
    if args.prompt and not args.interactive:
        async def run_single():
            try:
                _ = config.get_stream_url()
            except ValueError as e:
                console.print(f"[bold red]Configuration Error:[/bold red] {e}")
                console.print("[dim]Specify parameters via CLI flags or .env file.[/dim]")
                sys.exit(1)

            await execute_stream_turn(client, args.prompt, show_raw=args.raw)

        asyncio.run(run_single())
    else:
        try:
            _ = config.get_stream_url()
        except ValueError as e:
            console.print(f"[bold red]Configuration Warning:[/bold red] {e}")
            console.print("[dim]Configure GCP_PROJECT_ID and GCP_REASONING_ENGINE_ID in .env or via flags.[/dim]\n")

        asyncio.run(run_interactive_session(client, initial_prompt=args.prompt, show_raw=args.raw))


if __name__ == "__main__":
    main()
