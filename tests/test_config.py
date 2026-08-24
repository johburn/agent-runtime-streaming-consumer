"""Unit tests for AgentRuntimeConfig."""

import pytest
from agent_streaming_consumer.config import AgentRuntimeConfig
from agent_streaming_consumer.models import ClientMode, PayloadFormat, StreamingMode


def test_config_defaults():
    config = AgentRuntimeConfig(
        _env_file=None,
        project_id="test-project",
        reasoning_engine_id="123456789",
    )
    assert config.location == "us-central1"
    assert config.client_mode == ClientMode.VERTEX_SDK
    assert config.streaming_mode == StreamingMode.SSE
    assert config.payload_format == PayloadFormat.STANDARD
    assert config.user_id == "default-user"
    assert config.get_resource_name() == "projects/test-project/locations/us-central1/reasoningEngines/123456789"
    assert (
        config.get_stream_url()
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/reasoningEngines/123456789:streamQuery"
    )


def test_config_full_resource_path():
    full_path = "projects/my-prod-project/locations/us-east1/reasoningEngines/987654321"
    config = AgentRuntimeConfig(_env_file=None, reasoning_engine_id=full_path)
    assert config.get_resource_name() == full_path
    assert (
        config.get_stream_url()
        == f"https://us-east1-aiplatform.googleapis.com/v1/{full_path}:streamQuery"
    )


def test_config_missing_engine_id():
    config = AgentRuntimeConfig(_env_file=None, project_id="test-project")
    with pytest.raises(ValueError, match="Reasoning Engine ID is required"):
        config.get_resource_name()


def test_config_missing_project_id_with_short_engine():
    config = AgentRuntimeConfig(_env_file=None, reasoning_engine_id="12345")
    with pytest.raises(ValueError, match="GCP Project ID is required"):
        config.get_resource_name()


def test_config_endpoint_override():
    config = AgentRuntimeConfig(
        _env_file=None,
        project_id="p",
        reasoning_engine_id="123",
        api_endpoint_override="https://custom-proxy.internal/stream",
    )
    assert config.get_stream_url() == "https://custom-proxy.internal/stream"


def test_session_id_generation():
    config = AgentRuntimeConfig(_env_file=None, project_id="p", reasoning_engine_id="1")
    s1 = config.ensure_session_id()
    assert s1 is not None
    assert len(s1) > 10
    # Calling again returns same ID
    assert config.ensure_session_id() == s1
