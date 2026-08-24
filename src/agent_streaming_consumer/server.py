"""FastAPI Backend Server for Google Cloud Agent Engine / Agent Runtime SSE Streaming."""

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel

from .auth import GoogleAuthTokenProvider
from .client import AgentRuntimeClient
from .config import AgentRuntimeConfig
from .models import (
    AuthorTransfer,
    ClientMode,
    DoneEvent,
    ErrorEvent,
    RunConfig,
    StateDelta,
    StreamingMode,
    TextDelta,
    ThoughtDelta,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Default Google Cloud settings
DEFAULT_PROJECT_ID = "938422762731"
DEFAULT_LOCATION = "us-central1"
DEFAULT_ENGINE_ID = "projects/938422762731/locations/us-central1/reasoningEngines/6082567044333568000"

app = FastAPI(title="Google Cloud Agent Engine Streamer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    message: str
    engine_id: Optional[str] = None
    project_id: Optional[str] = DEFAULT_PROJECT_ID
    location: Optional[str] = DEFAULT_LOCATION
    user_id: Optional[str] = "user-123"
    session_id: Optional[str] = None
    streaming_mode: Optional[str] = "sse"  # "sse" or "none"


@app.get("/api/agents")
async def list_agents(
    project_id: str = Query(DEFAULT_PROJECT_ID),
    location: str = Query(DEFAULT_LOCATION),
):
    """Fetches list of reasoning engines from Vertex AI REST API."""
    try:
        auth = GoogleAuthTokenProvider()
        headers = auth.get_auth_headers()
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/reasoningEngines"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                engines = data.get("reasoningEngines", [])
                formatted = []
                for eng in engines:
                    res_name = eng.get("name", "")
                    short_id = res_name.split("/")[-1]
                    disp_name = eng.get("displayName") or short_id
                    desc = eng.get("description", "Sin descripción")
                    formatted.append({
                        "id": res_name,
                        "short_id": short_id,
                        "name": disp_name,
                        "description": desc,
                        "create_time": eng.get("createTime", ""),
                        "update_time": eng.get("updateTime", ""),
                    })
                return {"success": True, "agents": formatted}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                    "agents": [],
                }
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        return {"success": False, "error": str(e), "agents": []}


async def event_generator(req: StreamRequest) -> AsyncIterator[str]:
    """Generates SSE formatted events from Agent Runtime stream."""
    engine_id = req.engine_id or DEFAULT_ENGINE_ID
    mode = StreamingMode.SSE if req.streaming_mode == "sse" else StreamingMode.NONE

    config = AgentRuntimeConfig(
        project_id=req.project_id or DEFAULT_PROJECT_ID,
        location=req.location or DEFAULT_LOCATION,
        reasoning_engine_id=engine_id,
        client_mode=ClientMode.REST_API,
        streaming_mode=mode,
        run_config=RunConfig(streaming_mode=mode),
        user_id=req.user_id or "user-123",
        session_id=req.session_id,
    )
    client = AgentRuntimeClient(config=config)

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    chunk_count = 0
    accumulated_text = ""

    try:
        async for event in client.stream_query_api(message=req.message):
            chunk_count += 1
            t_now = time.perf_counter()

            event_payload: Dict[str, Any] = {
                "chunk_num": chunk_count,
                "timestamp": time.strftime("%H:%M:%S"),
                "elapsed": round(t_now - t0, 3),
                "author": event.author,
                "raw": event.raw_data or {},
            }

            if isinstance(event, TextDelta) and event.text:
                if ttft is None:
                    ttft = t_now - t0
                accumulated_text += event.text
                event_payload["event"] = "text"
                event_payload["text"] = event.text
                event_payload["accumulated_text"] = accumulated_text
                event_payload["ttft"] = round(ttft, 3)

            elif isinstance(event, ToolCall):
                event_payload["event"] = "tool_call"
                event_payload["tool_name"] = event.tool_name
                event_payload["arguments"] = event.arguments
                event_payload["call_id"] = event.call_id

            elif isinstance(event, ToolResult):
                event_payload["event"] = "tool_result"
                event_payload["tool_name"] = event.tool_name
                event_payload["output"] = event.output
                event_payload["call_id"] = event.call_id

            elif isinstance(event, ThoughtDelta):
                event_payload["event"] = "thought"
                event_payload["thought"] = event.thought

            elif isinstance(event, AuthorTransfer):
                event_payload["event"] = "author_transfer"
                event_payload["from_author"] = event.from_author
                event_payload["to_author"] = event.to_author

            elif isinstance(event, StateDelta):
                event_payload["event"] = "state_delta"
                event_payload["state_delta"] = event.state_delta

            elif isinstance(event, ErrorEvent):
                event_payload["event"] = "error"
                event_payload["error_message"] = event.error_message

            elif isinstance(event, DoneEvent):
                event_payload["event"] = "done"

            else:
                event_payload["event"] = "raw"

            yield f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"

        total_time = round(time.perf_counter() - t0, 3)
        done_payload = {
            "event": "completed",
            "ttft": round(ttft, 3) if ttft is not None else total_time,
            "total_time": total_time,
            "total_chunks": chunk_count,
            "total_text_length": len(accumulated_text),
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    except Exception as exc:
        err_payload = {
            "event": "error",
            "error_message": str(exc),
            "elapsed": round(time.perf_counter() - t0, 3),
        }
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"


@app.post("/api/stream")
async def stream_query(req: StreamRequest):
    """Streams SSE events to the browser."""
    return StreamingResponse(
        event_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve Static Assets
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    """Serves the main single-page UI."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Google Cloud Agent Engine Streamer</h1><p>Static files missing.</p>")
