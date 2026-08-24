"""Unit tests for AgentRuntimeSSEParser."""

import json
from agent_streaming_consumer.models import (
    AuthorTransfer,
    DoneEvent,
    ErrorEvent,
    StateDelta,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)
from agent_streaming_consumer.sse_parser import AgentRuntimeSSEParser


def test_parse_simple_text_content():
    parser = AgentRuntimeSSEParser()
    data = json.dumps({
        "content": {
            "role": "model",
            "parts": [{"text": "Hello, "}]
        }
    })
    events = parser.parse_sse_data(data)
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Hello, "
    assert events[0].accumulated_text == "Hello, "

    # Second token
    data2 = json.dumps({
        "content": {
            "role": "model",
            "parts": [{"text": "world!"}]
        }
    })
    events2 = parser.parse_sse_data(data2)
    assert len(events2) == 1
    assert events2[0].text == "world!"
    assert events2[0].accumulated_text == "Hello, world!"


def test_parse_direct_output_string():
    parser = AgentRuntimeSSEParser()
    data = json.dumps({"output": "Stream chunk"})
    events = parser.parse_sse_data(data)
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Stream chunk"


def test_parse_tool_call_and_result():
    parser = AgentRuntimeSSEParser()
    call_data = json.dumps({
        "content": {
            "role": "model",
            "parts": [{
                "function_call": {
                    "name": "get_weather",
                    "args": {"city": "London"},
                    "id": "fn_123"
                }
            }]
        }
    })
    events = parser.parse_sse_data(call_data)
    assert len(events) == 1
    assert isinstance(events[0], ToolCall)
    assert events[0].tool_name == "get_weather"
    assert events[0].arguments == {"city": "London"}
    assert events[0].call_id == "fn_123"

    res_data = json.dumps({
        "content": {
            "role": "tool",
            "parts": [{
                "function_response": {
                    "name": "get_weather",
                    "response": {"temp": "15C"}
                }
            }]
        }
    })
    events2 = parser.parse_sse_data(res_data)
    assert len(events2) == 1
    assert isinstance(events2[0], ToolResult)
    assert events2[0].tool_name == "get_weather"
    assert events2[0].output == {"temp": "15C"}


def test_parse_thought_delta():
    parser = AgentRuntimeSSEParser()
    thought_data = json.dumps({
        "content": {
            "parts": [{"thought": "Evaluating options..."}]
        }
    })
    events = parser.parse_sse_data(thought_data)
    assert len(events) == 1
    assert isinstance(events[0], ThoughtDelta)
    assert events[0].thought == "Evaluating options..."


def test_parse_author_transfer():
    parser = AgentRuntimeSSEParser()
    data1 = json.dumps({"author": "coordinator", "content": {"parts": [{"text": "Routing..."}]}})
    events1 = parser.parse_sse_data(data1)
    assert any(isinstance(e, AuthorTransfer) and e.to_author == "coordinator" for e in events1)

    data2 = json.dumps({"author": "specialist", "content": {"parts": [{"text": "Result"}]}})
    events2 = parser.parse_sse_data(data2)
    transfer = next(e for e in events2 if isinstance(e, AuthorTransfer))
    assert transfer.from_author == "coordinator"
    assert transfer.to_author == "specialist"


def test_parse_state_delta():
    parser = AgentRuntimeSSEParser()
    data = json.dumps({"actions": [{"state_delta": {"user_score": 10}}]})
    events = parser.parse_sse_data(data)
    assert len(events) == 1
    assert isinstance(events[0], StateDelta)
    assert events[0].state_delta == {"user_score": 10}


def test_parse_done_marker():
    parser = AgentRuntimeSSEParser()
    parser.accumulated_text = "Complete answer"
    events = parser.parse_sse_data("[DONE]")
    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)
    assert events[0].total_text == "Complete answer"
    assert parser.is_done is True


def test_parse_error_payload():
    parser = AgentRuntimeSSEParser()
    data = json.dumps({"error": {"code": 400, "message": "Invalid session ID"}})
    events = parser.parse_sse_data(data)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "Invalid session ID" in events[0].error_message
