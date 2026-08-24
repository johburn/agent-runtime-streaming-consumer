"""Example: Streaming remotely using native Google Cloud Vertex AI SDK."""

import asyncio
from agent_streaming_consumer import AgentRuntimeClient, AgentRuntimeConfig, ClientMode, TextDelta, ThoughtDelta, ToolCall


async def main():
    # Configure client to use native Vertex AI SDK
    config = AgentRuntimeConfig(client_mode=ClientMode.VERTEX_SDK)
    client = AgentRuntimeClient(config=config)

    print("=" * 60)
    print("Consuming Agent Runtime via Google Cloud Vertex AI SDK")
    print(f"Resource: {config.get_resource_name()}")
    print(f"Session ID: {client.session_id}")
    print("=" * 60)

    prompt = "Hello from Vertex AI SDK! List 3 key features of Vertex AI Reasoning Engine."
    print(f"\nUser: {prompt}\nAgent: ", end="", flush=True)

    async for event in client.stream_query_sdk(message=prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ThoughtDelta):
            print(f"\n[Reasoning]: {event.thought}", flush=True)
        elif isinstance(event, ToolCall):
            print(f"\n[Tool]: {event.tool_name}({event.arguments})", flush=True)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
