"""Unit tests for event models and RunConfig."""

from src.models import (
    AuthorTransfer,
    ClientMode,
    DoneEvent,
    ErrorEvent,
    EventType,
    RunConfig,
    StateDelta,
    StreamingMode,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)


def test_run_config():
    rc = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=50)
    d = rc.to_dict()
    assert d["streaming_mode"] == "sse"
    assert d["max_llm_calls"] == 50


def test_text_delta():
    event = TextDelta(text="Hello world")
    assert event.event_type == EventType.TEXT
    assert event.text == "Hello world"


def test_thought_delta():
    event = ThoughtDelta(thought="Let me think...")
    assert event.event_type == EventType.THOUGHT
    assert event.thought == "Let me think..."


def test_tool_call():
    event = ToolCall(tool_name="search", arguments={"query": "python"}, call_id="call_1")
    assert event.event_type == EventType.TOOL_CALL
    assert event.tool_name == "search"
    assert event.arguments == {"query": "python"}
    assert event.call_id == "call_1"


def test_tool_result():
    event = ToolResult(tool_name="search", output={"results": ["a", "b"]})
    assert event.event_type == EventType.TOOL_RESULT
    assert event.output == {"results": ["a", "b"]}


def test_author_transfer():
    event = AuthorTransfer(from_author="agent_a", to_author="agent_b")
    assert event.event_type == EventType.AUTHOR
    assert event.to_author == "agent_b"


def test_state_delta():
    event = StateDelta(state_delta={"cart": ["item1"]})
    assert event.event_type == EventType.STATE_DELTA
    assert event.state_delta == {"cart": ["item1"]}


def test_error_and_done_events():
    err = ErrorEvent(error_message="Something failed", status_code=500)
    assert err.event_type == EventType.ERROR
    assert err.status_code == 500

    done = DoneEvent(total_text="Full output")
    assert done.event_type == EventType.DONE
    assert done.total_text == "Full output"
