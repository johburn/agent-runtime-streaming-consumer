"""Data models, RunConfig and event schemas for Agent Runtime SSE streaming."""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClientMode(str, Enum):
    """Client invocation transport mode."""
    VERTEX_SDK = "sdk"    # Native Google Cloud Vertex AI SDK (ReasoningEngineExecutionClient)
    REST_API = "api"      # Direct Google Cloud :streamQuery REST API with SSE


class StreamingMode(str, Enum):
    """Streaming modes supported by ADK RunConfig."""
    NONE = "none"
    SSE = "sse"
    BIDI = "bidi"


class RunConfig(BaseModel):
    """Runtime configuration passed to ADK Runner / Agent Runtime."""
    streaming_mode: StreamingMode = StreamingMode.SSE
    max_llm_calls: Optional[int] = None
    include_thoughts_from_other_agents: Optional[bool] = None
    support_cfc: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Agent Runtime payload."""
        # In ADK, non-streaming mode is represented as null / None
        mode_val = None if self.streaming_mode == StreamingMode.NONE else self.streaming_mode.value
        data = {"streaming_mode": mode_val}
        if self.max_llm_calls is not None:
            data["max_llm_calls"] = self.max_llm_calls
        if self.include_thoughts_from_other_agents is not None:
            data["include_thoughts_from_other_agents"] = self.include_thoughts_from_other_agents
        if self.support_cfc is not None:
            data["support_cfc"] = self.support_cfc
        return data


class PayloadFormat(str, Enum):
    """Payload format structure sent to Agent Runtime."""
    STANDARD = "standard"       # {"input": {"message": ..., "user_id": ..., "session_id": ..., "run_config": ...}}
    ADK_PARTS = "adk_parts"     # {"input": {"new_message": {"role": "user", "parts": [{"text": ...}]}, ...}}
    RAW = "raw"                 # {"message": ..., "user_id": ..., "session_id": ...}


class EventType(str, Enum):
    """Types of streaming events parsed from Agent Runtime stream."""
    TEXT = "text"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_DELTA = "state_delta"
    AUTHOR = "author"
    ERROR = "error"
    DONE = "done"
    RAW = "raw"


class StreamEvent(BaseModel):
    """Base streaming event emitted by Agent Runtime."""
    event_type: EventType
    author: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    raw_data: Optional[Dict[str, Any]] = None


class TextDelta(StreamEvent):
    """Incremental text token / delta from the agent response."""
    event_type: EventType = EventType.TEXT
    text: str
    accumulated_text: Optional[str] = None


class ThoughtDelta(StreamEvent):
    """Agent reasoning or chain-of-thought delta."""
    event_type: EventType = EventType.THOUGHT
    thought: str


class ToolCall(StreamEvent):
    """Function / tool call triggered by the agent."""
    event_type: EventType = EventType.TOOL_CALL
    call_id: Optional[str] = None
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(StreamEvent):
    """Result returned after executing a tool."""
    event_type: EventType = EventType.TOOL_RESULT
    call_id: Optional[str] = None
    tool_name: Optional[str] = None
    output: Any = None


class StateDelta(StreamEvent):
    """Updates to the agent session state or memory."""
    event_type: EventType = EventType.STATE_DELTA
    state_delta: Dict[str, Any] = Field(default_factory=dict)


class AuthorTransfer(StreamEvent):
    """Multi-agent delegation / author change event."""
    event_type: EventType = EventType.AUTHOR
    from_author: Optional[str] = None
    to_author: str


class ErrorEvent(StreamEvent):
    """Error received from remote Agent Runtime or HTTP stream."""
    event_type: EventType = EventType.ERROR
    error_message: str
    status_code: Optional[int] = None


class DoneEvent(StreamEvent):
    """Stream completion event."""
    event_type: EventType = EventType.DONE
    total_text: str = ""
    finish_reason: Optional[str] = None


class RawEvent(StreamEvent):
    """Raw unparsed or unrecognized event payload."""
    event_type: EventType = EventType.RAW
    payload: Any = None
