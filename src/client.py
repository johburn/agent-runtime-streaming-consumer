"""Client for Google Cloud Agent Runtime / Agent Engine supporting Vertex AI SDK & streamQuery REST API."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, Optional
import httpx
from httpx_sse import aconnect_sse, connect_sse

from .auth import GoogleAuthTokenProvider
from .config import AgentRuntimeConfig
from .models import (
    ClientMode,
    DoneEvent,
    ErrorEvent,
    EventType,
    PayloadFormat,
    RunConfig,
    StreamEvent,
    StreamingMode,
    TextDelta,
)
from .sse_parser import AgentRuntimeSSEParser

logger = logging.getLogger(__name__)


class AgentRuntimeClient:
    """Client for querying Agent Runtime / Reasoning Engine agents via Vertex AI SDK or direct REST streamQuery API."""

    def __init__(
        self,
        config: Optional[AgentRuntimeConfig] = None,
        auth_provider: Optional[GoogleAuthTokenProvider] = None,
        **config_kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = AgentRuntimeConfig.from_deployment_metadata(**config_kwargs)

        self.auth_provider = auth_provider or GoogleAuthTokenProvider(
            credentials_path=self.config.credentials_path,
            access_token_override=self.config.access_token,
        )

        # Lazy reference to Vertex AI ReasoningEngine instance
        self._vertex_engine = None

    @property
    def session_id(self) -> Optional[str]:
        """Current session identifier."""
        return self.config.session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self.config.session_id = value

    def new_session(self) -> None:
        """Clears current session so the next query initializes a new session on Agent Runtime."""
        self.config.session_id = None

    def _get_vertex_engine(self):
        """Initializes Vertex AI SDK and loads the Reasoning Engine resource."""
        if self._vertex_engine is None:
            import vertexai
            from vertexai.preview import reasoning_engines

            credentials = self.auth_provider.get_credentials()
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location,
                credentials=credentials,
            )
            resource_name = self.config.get_resource_name()
            self._vertex_engine = reasoning_engines.ReasoningEngine(resource_name)
        return self._vertex_engine

    def build_payload(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """Builds the JSON request payload for Agent Runtime."""
        active_session = session_id if session_id is not None else self.config.session_id
        active_user = user_id or self.config.user_id
        active_run_config = run_config or self.config.run_config

        run_config_dict = active_run_config.to_dict() if active_run_config else {"streaming_mode": "sse"}

        input_data: Dict[str, Any] = {
            "user_id": active_user,
            "run_config": run_config_dict,
            **extra_kwargs,
        }

        if active_session:
            input_data["session_id"] = active_session

        if self.config.payload_format == PayloadFormat.ADK_PARTS:
            input_data["new_message"] = {
                "role": "user",
                "parts": [{"text": message}],
            }
            return {
                "class_method": "async_stream_query",
                "input": input_data,
            }
        elif self.config.payload_format == PayloadFormat.RAW:
            input_data["message"] = message
            return {
                "class_method": "async_stream_query",
                **input_data,
            }
        else:  # STANDARD
            input_data["message"] = message
            return {
                "class_method": "async_stream_query",
                "input": input_data,
            }

    async def stream_query_api(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        **extra_kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Streams responses via direct Google Cloud :streamQuery REST API using SSE."""
        url = self.config.get_stream_url()
        headers = self.auth_provider.get_auth_headers()
        payload = self.build_payload(
            message=message,
            session_id=session_id,
            user_id=user_id,
            run_config=run_config,
            **extra_kwargs,
        )

        parser = AgentRuntimeSSEParser()
        timeout = httpx.Timeout(self.config.timeout_seconds, connect=15.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        content = await response.aread()
                        error_text = content.decode("utf-8", errors="replace")
                        yield ErrorEvent(
                            error_message=f"Agent Runtime returned HTTP {response.status_code}: {error_text}",
                            status_code=response.status_code,
                        )
                        return

                    async for line in response.aiter_lines():
                        if line:
                            raw_line = line[6:] if line.startswith("data: ") else line
                            for event in parser.parse_sse_data(raw_line):
                                yield event

            except httpx.HTTPStatusError as exc:
                yield ErrorEvent(
                    error_message=f"HTTP error {exc.response.status_code}: {exc.response.text}",
                    status_code=exc.response.status_code,
                )
                return
            except Exception as exc:
                yield ErrorEvent(error_message=f"Stream connection error: {str(exc)}")
                return

        if not parser.is_done:
            yield parser.finalize()

    async def stream_query_sdk(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        **extra_kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Streams responses using native Google Cloud Vertex AI SDK."""
        parser = AgentRuntimeSSEParser()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        try:
            engine = self._get_vertex_engine()
            payload = self.build_payload(
                message=message,
                session_id=session_id,
                user_id=user_id,
                run_config=run_config,
                **extra_kwargs,
            )

            request = {
                "name": engine.resource_name,
                "class_method": "async_stream_query",
                "input": payload.get("input", payload),
            }

            def _grpc_worker():
                try:
                    stream = engine.execution_api_client.stream_query_reasoning_engine(request=request)
                    for response in stream:
                        loop.call_soon_threadsafe(queue.put_nowait, response)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            # Start worker in background executor thread
            loop.run_in_executor(None, _grpc_worker)

            while True:
                response = await queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    yield ErrorEvent(error_message=f"Vertex AI SDK streaming error: {str(response)}")
                    return

                if hasattr(response, "data") and response.data:
                    raw_text = response.data.decode("utf-8", errors="replace") if isinstance(response.data, bytes) else str(response.data)
                    for event in parser.parse_sse_data(raw_text):
                        yield event
                elif isinstance(response, dict):
                    for event in parser.parse_payload_dict(response):
                        yield event
                elif isinstance(response, str):
                    for event in parser.parse_sse_data(response):
                        yield event

        except Exception as exc:
            yield ErrorEvent(error_message=f"Vertex AI SDK streaming error: {str(exc)}")
            return

        if not parser.is_done:
            yield parser.finalize()

    async def stream_query(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Streams responses using configured backend (Vertex AI SDK or REST API)."""
        active_mode = mode or self.config.client_mode
        if active_mode == ClientMode.VERTEX_SDK:
            async for event in self.stream_query_sdk(
                message=message,
                session_id=session_id,
                user_id=user_id,
                run_config=run_config,
                **extra_kwargs,
            ):
                yield event
        else:
            async for event in self.stream_query_api(
                message=message,
                session_id=session_id,
                user_id=user_id,
                run_config=run_config,
                **extra_kwargs,
            ):
                yield event

    async def stream_text(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> AsyncIterator[str]:
        """Convenience generator that yields only the text tokens."""
        async for event in self.stream_query(
            message=message,
            session_id=session_id,
            user_id=user_id,
            run_config=run_config,
            mode=mode,
            **extra_kwargs,
        ):
            if isinstance(event, TextDelta):
                yield event.text

    async def query(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> str:
        """Collects and returns the full completed agent response as a string."""
        full_text = []
        async for text_chunk in self.stream_text(
            message=message,
            session_id=session_id,
            user_id=user_id,
            run_config=run_config,
            mode=mode,
            **extra_kwargs,
        ):
            full_text.append(text_chunk)
        return "".join(full_text)

    def stream_query_sync(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> Iterator[StreamEvent]:
        """Synchronously streams responses from Agent Runtime."""
        active_mode = mode or self.config.client_mode
        parser = AgentRuntimeSSEParser()

        if active_mode == ClientMode.VERTEX_SDK:
            try:
                engine = self._get_vertex_engine()
                payload = self.build_payload(
                    message=message,
                    session_id=session_id,
                    user_id=user_id,
                    run_config=run_config,
                    **extra_kwargs,
                )
                request = {
                    "name": engine.resource_name,
                    "class_method": "async_stream_query",
                    "input": payload.get("input", payload),
                }
                stream = engine.execution_api_client.stream_query_reasoning_engine(request=request)
                for response in stream:
                    if hasattr(response, "data") and response.data:
                        raw_text = response.data.decode("utf-8", errors="replace") if isinstance(response.data, bytes) else str(response.data)
                        for event in parser.parse_sse_data(raw_text):
                            yield event
                    elif isinstance(response, dict):
                        for event in parser.parse_payload_dict(response):
                            yield event
                    elif isinstance(response, str):
                        for event in parser.parse_sse_data(response):
                            yield event
            except Exception as exc:
                yield ErrorEvent(error_message=f"Vertex AI SDK error: {str(exc)}")
                return
        else:  # REST_API
            url = self.config.get_stream_url()
            headers = self.auth_provider.get_auth_headers()
            payload = self.build_payload(
                message=message,
                session_id=session_id,
                user_id=user_id,
                run_config=run_config,
                **extra_kwargs,
            )

            timeout = httpx.Timeout(self.config.timeout_seconds, connect=15.0)

            with httpx.Client(timeout=timeout) as client:
                try:
                    with client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            content = response.read()
                            error_text = content.decode("utf-8", errors="replace")
                            yield ErrorEvent(
                                error_message=f"Agent Runtime returned HTTP {response.status_code}: {error_text}",
                                status_code=response.status_code,
                            )
                            return

                        for line in response.iter_lines():
                            if line:
                                raw_line = line[6:] if line.startswith("data: ") else line
                                for event in parser.parse_sse_data(raw_line):
                                    yield event

                except httpx.HTTPStatusError as exc:
                    yield ErrorEvent(
                        error_message=f"HTTP error {exc.response.status_code}: {exc.response.text}",
                        status_code=exc.response.status_code,
                    )
                    return
                except Exception as exc:
                    yield ErrorEvent(error_message=f"Stream connection error: {str(exc)}")
                    return

        if not parser.is_done:
            yield parser.finalize()

    def stream_text_sync(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> Iterator[str]:
        """Synchronous convenience generator that yields only the text tokens."""
        for event in self.stream_query_sync(
            message=message,
            session_id=session_id,
            user_id=user_id,
            run_config=run_config,
            mode=mode,
            **extra_kwargs,
        ):
            if isinstance(event, TextDelta):
                yield event.text

    def query_sync(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        mode: Optional[ClientMode] = None,
        **extra_kwargs,
    ) -> str:
        """Synchronously collects and returns the full completed agent response as a string."""
        full_text = []
        for text_chunk in self.stream_text_sync(
            message=message,
            session_id=session_id,
            user_id=user_id,
            run_config=run_config,
            mode=mode,
            **extra_kwargs,
        ):
            full_text.append(text_chunk)
        return "".join(full_text)
