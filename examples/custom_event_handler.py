"""Example: Detailed handling of distinct SSE event types."""

import asyncio
from agent_streaming_consumer import (
    AgentRuntimeClient,
    AgentRuntimeConfig,
    AuthorTransfer,
    DoneEvent,
    ErrorEvent,
    StateDelta,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)


async def handle_agent_events(prompt: str):
    config = AgentRuntimeConfig()
    client = AgentRuntimeClient(config=config)

    print(f"Prompt: {prompt}")
    print("=" * 60)

    async for event in client.stream_query(message=prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

        elif isinstance(event, ThoughtDelta):
            print(f"\n[Thought / Reasoning]: {event.thought}")

        elif isinstance(event, ToolCall):
            print(f"\n[Tool Executing]: {event.tool_name} with args: {event.arguments}")

        elif isinstance(event, ToolResult):
            print(f"\n[Tool Output]: {event.output}")

        elif isinstance(event, AuthorTransfer):
            print(f"\n[Agent Switched]: {event.from_author} -> {event.to_author}")

        elif isinstance(event, StateDelta):
            print(f"\n[State Updated]: {event.state_delta}")

        elif isinstance(event, ErrorEvent):
            print(f"\n[Error]: {event.error_message}")

        elif isinstance(event, DoneEvent):
            print(f"\n\n[Stream Completed] - Total response length: {len(event.total_text)} chars")


if __name__ == "__main__":
    asyncio.run(handle_agent_events("Check if 2024 is a leap year and explain why."))
