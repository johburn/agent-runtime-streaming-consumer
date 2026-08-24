"""Example: Multi-turn conversation maintaining persistent session state."""

import asyncio
from agent_streaming_consumer import AgentRuntimeClient, AgentRuntimeConfig, TextDelta


async def main():
    config = AgentRuntimeConfig()
    client = AgentRuntimeClient(config=config)

    print(f"Starting Multi-Turn Conversation (Session: {client.session_id})")
    print("-" * 60)

    # Turn 1
    turn1_prompt = "My favorite programming language is Python."
    print(f"\nTurn 1 User: {turn1_prompt}\nAgent: ", end="", flush=True)
    async for event in client.stream_query(message=turn1_prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

    # Turn 2 - Agent remembers context from Turn 1 due to the same session_id
    turn2_prompt = "What did I say was my favorite language?"
    print(f"\n\nTurn 2 User: {turn2_prompt}\nAgent: ", end="", flush=True)
    async for event in client.stream_query(message=turn2_prompt):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

    print("\n\n" + "-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
