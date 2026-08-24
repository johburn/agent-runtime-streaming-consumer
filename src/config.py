"""Configuration management for Agent Runtime SSE Streaming Consumer."""

import json
import os
import re
import uuid
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import ClientMode, PayloadFormat, RunConfig, StreamingMode


class AgentRuntimeConfig(BaseSettings):
    """Settings and configuration for connecting to Google Cloud Agent Runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Invocation backend mode: 'sdk' (Vertex AI SDK) or 'api' (Direct REST streamQuery URL)
    client_mode: ClientMode = Field(
        default=ClientMode.VERTEX_SDK,
        validation_alias=AliasChoices("client_mode", "CLIENT_MODE"),
        description="Invocation mode: 'sdk' for Vertex AI SDK, 'api' for direct REST streamQuery URL",
    )

    # GCP Project and Location
    project_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("project_id", "GCP_PROJECT_ID"),
        description="Google Cloud project ID",
    )
    location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("location", "GCP_LOCATION"),
        description="Google Cloud location / region (e.g. us-central1, us-east1)",
    )

    # Reasoning Engine / Agent Runtime ID
    reasoning_engine_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("reasoning_engine_id", "GCP_REASONING_ENGINE_ID"),
        description="Reasoning Engine resource ID, name or full resource path",
    )

    # Optional custom API endpoint (e.g. for testing, mocks, or private VPC proxy)
    api_endpoint_override: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("api_endpoint_override", "AGENT_RUNTIME_ENDPOINT_OVERRIDE"),
        description="Direct endpoint URL override",
    )

    # Streaming & Run Configuration (defaults to SSE)
    streaming_mode: StreamingMode = Field(
        default=StreamingMode.SSE,
        validation_alias=AliasChoices("streaming_mode", "STREAMING_MODE"),
        description="Streaming mode (default: sse)",
    )
    payload_format: PayloadFormat = Field(
        default=PayloadFormat.STANDARD,
        validation_alias=AliasChoices("payload_format", "PAYLOAD_FORMAT"),
        description="Payload serialization structure sent to Agent Runtime",
    )
    run_config: RunConfig = Field(
        default_factory=lambda: RunConfig(streaming_mode=StreamingMode.SSE),
        validation_alias=AliasChoices("run_config"),
        description="ADK RunConfig controlling streaming mode and limits",
    )

    # Session & User identification
    user_id: str = Field(
        default="default-user",
        validation_alias=AliasChoices("user_id", "AGENT_USER_ID"),
        description="User identifier passed to the agent",
    )
    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "AGENT_SESSION_ID"),
        description="Session identifier for multi-turn conversation (None creates a new server session)",
    )

    # Network & Auth
    timeout_seconds: float = Field(
        default=180.0,
        validation_alias=AliasChoices("timeout_seconds", "HTTP_TIMEOUT_SECONDS"),
        description="HTTP request timeout in seconds",
    )
    credentials_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("credentials_path", "GOOGLE_APPLICATION_CREDENTIALS"),
        description="Path to service account JSON key file (if not using ADC)",
    )
    access_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("access_token", "GCP_ACCESS_TOKEN"),
        description="Explicit OAuth2 bearer token override",
    )

    def ensure_session_id(self) -> str:
        """Returns existing session_id or generates a new random UUID session_id."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self.session_id

    def get_resource_name(self) -> str:
        """Resolves the full GCP resource name for the reasoning engine."""
        if not self.reasoning_engine_id:
            raise ValueError(
                "Reasoning Engine ID is required. Set GCP_REASONING_ENGINE_ID in .env or pass reasoning_engine_id."
            )

        engine = self.reasoning_engine_id.strip()

        # If already a full resource path
        if engine.startswith("projects/"):
            return engine

        if not self.project_id:
            raise ValueError(
                "GCP Project ID is required when providing a short engine ID. Set GCP_PROJECT_ID in .env."
            )

        return f"projects/{self.project_id}/locations/{self.location}/reasoningEngines/{engine}"

    def get_stream_url(self) -> str:
        """Constructs the HTTP POST SSE streaming URL for Agent Runtime :streamQuery."""
        if self.api_endpoint_override:
            return self.api_endpoint_override

        resource_name = self.get_resource_name()

        # Extract location from resource name if present
        match = re.search(r"locations/([^/]+)/", resource_name)
        loc = match.group(1) if match else self.location

        base_host = f"https://{loc}-aiplatform.googleapis.com/v1"
        return f"{base_host}/{resource_name}:streamQuery"

    @classmethod
    def from_deployment_metadata(
        cls, metadata_path: str = "deployment_metadata.json", **kwargs
    ) -> "AgentRuntimeConfig":
        """Loads configuration automatically from deployment_metadata.json if present."""
        config_data = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    remote_id = meta.get("remote_agent_runtime_id")
                    if remote_id:
                        config_data["reasoning_engine_id"] = remote_id
                        match = re.search(
                            r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/([^/]+)",
                            remote_id,
                        )
                        if match:
                            config_data["project_id"] = match.group(1)
                            config_data["location"] = match.group(2)
            except Exception:
                pass

        config_data.update({k: v for k, v in kwargs.items() if v is not None})
        return cls(**config_data)
