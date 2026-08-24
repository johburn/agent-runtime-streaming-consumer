"""Example: Streaming remotely using direct Google Cloud streamQuery REST URL with SSE."""

import asyncio
from src import AgentRuntimeClient, AgentRuntimeConfig, ClientMode, TextDelta, ThoughtDelta, ToolCall


async def main():
    # Configure client to use direct REST API
    config = AgentRuntimeConfig(client_mode=ClientMode.REST_API)
    client = AgentRuntimeClient(config=config)

    print("=" * 60)
    print("Consuming Agent Runtime via direct :streamQuery REST API")
    print(f"Stream URL: {config.get_stream_url()}")
    print(f"Session ID: {client.session_id}")
    print("=" * 60)

    prompt = "Hello via direct REST SSE! Give me a quick summary of Server-Sent Events."
    print(f"\nUser: {prompt}\nAgent: ", end="", flush=True)

    async for event in client.stream_query_api(message=prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ThoughtDelta):
            print(f"\n[Reasoning]: {event.thought}", flush=True)
        elif isinstance(event, ToolCall):
            print(f"\n[Tool]: {event.tool_name}({event.arguments})", flush=True)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
