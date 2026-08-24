"""Authentication handling and token management for Google Cloud APIs."""

from typing import Dict, Optional
import google.auth
import google.auth.credentials
import google.auth.transport.requests
from google.oauth2 import service_account


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GoogleAuthTokenProvider:
    """Provides and automatically refreshes Google Cloud OAuth2 credentials and bearer tokens."""

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        access_token_override: Optional[str] = None,
        scopes: Optional[list] = None,
    ):
        self.access_token_override = access_token_override
        self.credentials_path = credentials_path
        self.scopes = scopes or [CLOUD_PLATFORM_SCOPE]
        self._credentials: Optional[google.auth.credentials.Credentials] = None
        self._auth_request = google.auth.transport.requests.Request()
        self._init_credentials()

    def _init_credentials(self) -> None:
        """Initializes credentials from service account file or Application Default Credentials (ADC)."""
        if self.access_token_override:
            return

        if self.credentials_path:
            self._credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.scopes,
            )
        else:
            self._credentials, _ = google.auth.default(scopes=self.scopes)

    def get_credentials(self) -> Optional[google.auth.credentials.Credentials]:
        """Returns the underlying google.auth Credentials object for Vertex AI SDK."""
        if not self._credentials and not self.access_token_override:
            self._init_credentials()
        return self._credentials

    def get_token(self) -> str:
        """Returns a valid OAuth2 access token, refreshing if necessary."""
        if self.access_token_override:
            return self.access_token_override

        if not self._credentials:
            self._init_credentials()

        if not self._credentials:
            raise RuntimeError("Failed to initialize Google Cloud credentials.")

        # Refresh if not valid or expired
        if not self._credentials.valid:
            self._credentials.refresh(self._auth_request)

        return self._credentials.token

    def get_auth_headers(self) -> Dict[str, str]:
        """Returns HTTP headers including Authorization Bearer token."""
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
