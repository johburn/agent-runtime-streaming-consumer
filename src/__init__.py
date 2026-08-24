"""Google Cloud Agent Runtime / Agent Engine SSE Streaming Consumer Client Library."""

from .auth import GoogleAuthTokenProvider
from .client import AgentRuntimeClient
from .config import AgentRuntimeConfig
from .models import (
    AuthorTransfer,
    ClientMode,
    DoneEvent,
    ErrorEvent,
    EventType,
    PayloadFormat,
    RawEvent,
    RunConfig,
    StateDelta,
    StreamEvent,
    StreamingMode,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)
from .sse_parser import AgentRuntimeSSEParser

__version__ = "0.1.0"

__all__ = [
    "AgentRuntimeClient",
    "AgentRuntimeConfig",
    "GoogleAuthTokenProvider",
    "AgentRuntimeSSEParser",
    "ClientMode",
    "RunConfig",
    "StreamingMode",
    "PayloadFormat",
    "EventType",
    "StreamEvent",
    "TextDelta",
    "ThoughtDelta",
    "ToolCall",
    "ToolResult",
    "StateDelta",
    "AuthorTransfer",
    "ErrorEvent",
    "DoneEvent",
    "RawEvent",
]
