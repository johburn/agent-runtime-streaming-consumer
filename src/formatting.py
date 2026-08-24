"""Rich console formatters and visual renderers for Agent Runtime events."""

import json
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .models import (
    AuthorTransfer,
    DoneEvent,
    ErrorEvent,
    StateDelta,
    StreamEvent,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)


def format_json_safe(data: Any) -> str:
    """Pretty formats data as JSON if possible."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)


def render_event(event: StreamEvent, console: Console) -> None:
    """Renders a StreamEvent to the Rich console with appropriate styling."""
    if isinstance(event, TextDelta):
        console.print(event.text, end="")

    elif isinstance(event, ThoughtDelta):
        thought_text = Text(f"\n🧠 Reasoning: {event.thought}", style="italic dim cyan")
        console.print(thought_text)

    elif isinstance(event, ToolCall):
        args_syntax = Syntax(
            format_json_safe(event.arguments),
            "json",
            theme="monokai",
            line_numbers=False,
        )
        title = f"🔧 Tool Call: [bold yellow]{event.tool_name}[/bold yellow]"
        if event.call_id:
            title += f" ([dim]{event.call_id}[/dim])"
        panel = Panel(
            args_syntax,
            title=title,
            border_style="yellow",
            expand=False,
        )
        console.print(panel)

    elif isinstance(event, ToolResult):
        output_str = format_json_safe(event.output)
        out_syntax = Syntax(
            output_str,
            "json" if output_str.startswith(("{", "[")) else "text",
            theme="monokai",
            line_numbers=False,
        )
        title = f"✅ Tool Result: [bold green]{event.tool_name or 'Output'}[/bold green]"
        panel = Panel(
            out_syntax,
            title=title,
            border_style="green",
            expand=False,
        )
        console.print(panel)

    elif isinstance(event, AuthorTransfer):
        from_str = f"[dim]{event.from_author}[/dim] ➔ " if event.from_author else ""
        msg = f"🤖 Agent Delegation: {from_str}[bold cyan]{event.to_author}[/bold cyan]"
        console.print(Panel(msg, border_style="cyan", expand=False))

    elif isinstance(event, StateDelta):
        state_syntax = Syntax(
            format_json_safe(event.state_delta),
            "json",
            theme="monokai",
            line_numbers=False,
        )
        panel = Panel(
            state_syntax,
            title="💾 State / Memory Update",
            border_style="magenta",
            expand=False,
        )
        console.print(panel)

    elif isinstance(event, ErrorEvent):
        err_msg = f"[bold red]Error[/bold red]: {event.error_message}"
        if event.status_code:
            err_msg += f" (Status: {event.status_code})"
        console.print(Panel(err_msg, border_style="red", expand=False))
