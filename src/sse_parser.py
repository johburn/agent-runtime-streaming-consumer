"""Server-Sent Events (SSE) parser for Google Cloud Agent Runtime / Agent Engine streams."""

import json
import logging
from typing import Any, Dict, List, Optional

from .models import (
    AuthorTransfer,
    DoneEvent,
    ErrorEvent,
    EventType,
    RawEvent,
    StateDelta,
    StreamEvent,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


class AgentRuntimeSSEParser:
    """Parses SSE event chunks and transforms them into strongly-typed StreamEvents."""

    def __init__(self):
        self.accumulated_text: str = ""
        self.current_author: Optional[str] = None
        self.is_done: bool = False
        self._received_partial_text: bool = False

    def parse_sse_data(self, data_str: str) -> List[StreamEvent]:
        """Parses a raw SSE 'data:' payload string into one or more StreamEvents."""
        data_str = data_str.strip()
        if not data_str:
            return []

        # Handle stream completion markers
        if data_str in ("[DONE]", "DONE"):
            self.is_done = True
            return [DoneEvent(total_text=self.accumulated_text)]

        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            # If payload is plain text
            self.accumulated_text += data_str
            return [
                TextDelta(
                    text=data_str,
                    accumulated_text=self.accumulated_text,
                    author=self.current_author,
                )
            ]

        return self.parse_payload_dict(payload)

    def parse_payload_dict(self, payload: Any) -> List[StreamEvent]:
        """Parses a decoded JSON object or list into StreamEvents."""
        events: List[StreamEvent] = []

        if isinstance(payload, list):
            for item in payload:
                events.extend(self.parse_payload_dict(item))
            return events

        if not isinstance(payload, dict):
            text_val = str(payload)
            self.accumulated_text += text_val
            return [
                TextDelta(
                    text=text_val,
                    accumulated_text=self.accumulated_text,
                    author=self.current_author,
                )
            ]

        # 1. Check for errors
        if "error" in payload:
            err = payload["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            events.append(ErrorEvent(error_message=msg, status_code=code, raw_data=payload))
            return events

        # 2. Check for author transfer / delegation
        author = payload.get("author") or payload.get("agent_name")
        if author and author != self.current_author:
            events.append(
                AuthorTransfer(
                    from_author=self.current_author,
                    to_author=author,
                    raw_data=payload,
                )
            )
            self.current_author = author

        # 3. Check for direct Reasoning Engine 'output' wrapping
        if "output" in payload:
            output_data = payload["output"]
            if isinstance(output_data, str):
                self.accumulated_text += output_data
                events.append(
                    TextDelta(
                        text=output_data,
                        accumulated_text=self.accumulated_text,
                        author=self.current_author,
                        raw_data=payload,
                    )
                )
                return events
            elif isinstance(output_data, dict):
                inner_events = self.parse_payload_dict(output_data)
                if inner_events:
                    return inner_events

        # 4. Check for ADK content / parts structure
        is_partial = payload.get("partial", True)
        content = payload.get("content")
        if isinstance(content, dict):
            parts = content.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    # Text part
                    if "text" in part and part["text"]:
                        text_chunk = part["text"]
                        # Avoid duplicating full message if partial: false arrives after partial chunks
                        if not is_partial and self._received_partial_text:
                            pass
                        else:
                            if is_partial:
                                self._received_partial_text = True
                            self.accumulated_text += text_chunk
                            events.append(
                                TextDelta(
                                    text=text_chunk,
                                    accumulated_text=self.accumulated_text,
                                    author=self.current_author,
                                    raw_data=payload,
                                )
                            )

                    # Thought / reasoning part
                    if "thought" in part and part["thought"]:
                        events.append(
                            ThoughtDelta(
                                thought=part["thought"],
                                author=self.current_author,
                                raw_data=payload,
                            )
                        )

                    # Function / Tool call
                    if "function_call" in part:
                        fc = part["function_call"]
                        events.append(
                            ToolCall(
                                tool_name=fc.get("name", "unknown"),
                                arguments=fc.get("args", {}),
                                call_id=fc.get("id"),
                                author=self.current_author,
                                raw_data=payload,
                            )
                        )

                    # Function / Tool response
                    if "function_response" in part:
                        fr = part["function_response"]
                        events.append(
                            ToolResult(
                                tool_name=fr.get("name"),
                                output=fr.get("response", fr),
                                call_id=fr.get("id"),
                                author=self.current_author,
                                raw_data=payload,
                            )
                        )

        # 5. Direct top-level fields
        if "thought" in payload:
            events.append(
                ThoughtDelta(
                    thought=str(payload["thought"]),
                    author=self.current_author,
                    raw_data=payload,
                )
            )

        if "call" in payload or "tool_call" in payload:
            call_obj = payload.get("call") or payload.get("tool_call")
            if isinstance(call_obj, dict):
                events.append(
                    ToolCall(
                        tool_name=call_obj.get("name", "unknown"),
                        arguments=call_obj.get("args", {}),
                        call_id=call_obj.get("id"),
                        author=self.current_author,
                        raw_data=payload,
                    )
                )

        if "tool_response" in payload:
            tr = payload["tool_response"]
            if isinstance(tr, dict):
                events.append(
                    ToolResult(
                        tool_name=tr.get("name"),
                        output=tr.get("output", tr),
                        call_id=tr.get("id"),
                        author=self.current_author,
                        raw_data=payload,
                    )
                )

        # 6. Check for state deltas or action events
        actions = payload.get("actions", [])
        if isinstance(actions, list):
            for act in actions:
                if isinstance(act, dict) and "state_delta" in act:
                    events.append(
                        StateDelta(
                            state_delta=act["state_delta"],
                            author=self.current_author,
                            raw_data=payload,
                        )
                    )

        if "state_delta" in payload:
            events.append(
                StateDelta(
                    state_delta=payload["state_delta"],
                    author=self.current_author,
                    raw_data=payload,
                )
            )

        return events

    def finalize(self) -> DoneEvent:
        """Produces the final DoneEvent if not already finished."""
        self.is_done = True
        return DoneEvent(total_text=self.accumulated_text, author=self.current_author)
