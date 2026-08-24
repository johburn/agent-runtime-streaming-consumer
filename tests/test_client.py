"""Integration / mock tests for AgentRuntimeClient."""

from unittest.mock import MagicMock
import pytest
import respx
from httpx import Response

from agent_streaming_consumer.auth import GoogleAuthTokenProvider
from agent_streaming_consumer.client import AgentRuntimeClient
from agent_streaming_consumer.config import AgentRuntimeConfig
from agent_streaming_consumer.models import ClientMode, DoneEvent, TextDelta


@pytest.fixture
def mock_rest_client():
    config = AgentRuntimeConfig(
        _env_file=None,
        client_mode=ClientMode.REST_API,
        project_id="my-project",
        location="us-central1",
        reasoning_engine_id="12345",
    )
    auth = GoogleAuthTokenProvider(access_token_override="mock-test-token")
    return AgentRuntimeClient(config=config, auth_provider=auth)


@pytest.mark.asyncio
@respx.mock
async def test_async_stream_query_rest(mock_rest_client):
    stream_url = mock_rest_client.config.get_stream_url()

    sse_body = (
        'data: {"content": {"role": "model", "parts": [{"text": "Hello "}]}}\n\n'
        'data: {"content": {"role": "model", "parts": [{"text": "from Agent Runtime!"}]}}\n\n'
        'data: [DONE]\n\n'
    )

    respx.post(stream_url).mock(
        return_value=Response(
            200,
            content=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    events = []
    async for event in mock_rest_client.stream_query(message="Hi"):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_deltas) == 2
    assert text_deltas[0].text == "Hello "
    assert text_deltas[1].text == "from Agent Runtime!"

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].total_text == "Hello from Agent Runtime!"


@pytest.mark.asyncio
@respx.mock
async def test_async_query_helper(mock_rest_client):
    stream_url = mock_rest_client.config.get_stream_url()

    sse_body = (
        'data: {"output": "Direct reasoning engine text response"}\n\n'
        'data: [DONE]\n\n'
    )

    respx.post(stream_url).mock(
        return_value=Response(
            200,
            content=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    result = await mock_rest_client.query(message="Hi")
    assert result == "Direct reasoning engine text response"


@respx.mock
def test_sync_stream_query_rest(mock_rest_client):
    stream_url = mock_rest_client.config.get_stream_url()

    sse_body = (
        'data: {"content": {"role": "model", "parts": [{"text": "Sync stream worked!"}]}}\n\n'
        'data: [DONE]\n\n'
    )

    respx.post(stream_url).mock(
        return_value=Response(
            200,
            content=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    result = mock_rest_client.query_sync(message="Hi")
    assert result == "Sync stream worked!"


@pytest.mark.asyncio
async def test_stream_query_vertex_sdk_mock():
    config = AgentRuntimeConfig(
        _env_file=None,
        client_mode=ClientMode.VERTEX_SDK,
        project_id="my-project",
        location="us-central1",
        reasoning_engine_id="projects/my-project/locations/us-central1/reasoningEngines/12345",
    )
    auth = GoogleAuthTokenProvider(access_token_override="mock-test-token")
    client = AgentRuntimeClient(config=config, auth_provider=auth)

    mock_engine = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = '{"content": {"role": "model", "parts": [{"text": "Vertex SDK Stream"}]}}'
    mock_engine.execution_api_client.stream_query_reasoning_engine.return_value = [mock_resp]
    mock_engine.resource_name = "projects/my-project/locations/us-central1/reasoningEngines/12345"
    client._vertex_engine = mock_engine

    events = []
    async for event in client.stream_query(message="Hello"):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_deltas) == 1
    assert text_deltas[0].text == "Vertex SDK Stream"
