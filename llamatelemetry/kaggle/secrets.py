"""
llamatelemetry.kaggle.secrets - Auto-load Kaggle secrets.

Replaces repetitive secrets handling code:

Before:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token

After:
    secrets = auto_load_secrets()
    hf_token = secrets.get("HF_TOKEN")  # Already in environment
"""

from typing import Dict, Optional, List
import os


class KaggleSecrets:
    """
    Wrapper for Kaggle secrets with caching and fallback.

    Automatically detects Kaggle environment and loads secrets from
    the Kaggle secrets API. Falls back to environment variables on
    non-Kaggle platforms.

    Example:
        >>> secrets = KaggleSecrets()
        >>> hf_token = secrets.get("HF_TOKEN")
        >>> if hf_token:
        ...     print("HuggingFace token loaded")
    """

    # Common secret names used in llamatelemetry notebooks
    KNOWN_SECRETS = [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "Graphistry_Personal_Key_ID",
        "Graphistry_Personal_Secret_Key",
        "GRAPHISTRY_USERNAME",
        "GRAPHISTRY_PASSWORD",
        "GRAPHISTRY_API_KEY",
        "OTLP_ENDPOINT",
        "OTLP_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]

    def __init__(self, auto_load: bool = True):
        """
        Initialize KaggleSecrets.

        Args:
            auto_load: Automatically load all known secrets on init
        """
        self._secrets: Dict[str, str] = {}
        self._client = None
        self._is_kaggle = self._detect_kaggle()

        if auto_load:
            self._load_all()

    @property
    def is_kaggle(self) -> bool:
        """Check if running on Kaggle."""
        return self._is_kaggle

    def _detect_kaggle(self) -> bool:
        """Detect if running on Kaggle."""
        return (
            "KAGGLE_KERNEL_RUN_TYPE" in os.environ or
            "KAGGLE_URL_BASE" in os.environ or
            os.path.exists("/kaggle")
        )

    def _load_all(self):
        """Load all known secrets."""
        if not self._is_kaggle:
            # Fall back to environment variables
            for name in self.KNOWN_SECRETS:
                value = os.environ.get(name)
                if value:
                    self._secrets[name] = value
            return

        # Try Kaggle secrets API
        try:
            from kaggle_secrets import UserSecretsClient
            self._client = UserSecretsClient()

            for name in self.KNOWN_SECRETS:
                try:
                    value = self._client.get_secret(name)
                    if value:
                        self._secrets[name] = value
                        # Also set in environment for libraries that expect it
                        os.environ[name] = value
                except Exception:
                    pass  # Secret not set
        except ImportError:
            # kaggle_secrets not available, use env vars
            for name in self.KNOWN_SECRETS:
                value = os.environ.get(name)
                if value:
                    self._secrets[name] = value

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by name.

        Args:
            name: Secret name
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if name in self._secrets:
            return self._secrets[name]

        # Try environment variable
        value = os.environ.get(name)
        if value:
            self._secrets[name] = value
            return value

        # Try Kaggle API
        if self._client:
            try:
                value = self._client.get_secret(name)
                if value:
                    self._secrets[name] = value
                    os.environ[name] = value
                    return value
            except Exception:
                pass

        return default

    def set_in_env(self, name: str) -> bool:
        """
        Ensure a secret is set in the environment.

        Args:
            name: Secret name to set in environment

        Returns:
            True if secret was found and set
        """
        value = self.get(name)
        if value:
            os.environ[name] = value
            return True
        return False

    def get_all(self) -> Dict[str, str]:
        """
        Get all loaded secrets.

        Returns:
            Dictionary of secret name to value
        """
        return self._secrets.copy()

    def list_available(self) -> List[str]:
        """
        List names of available secrets.

        Returns:
            List of secret names that have values
        """
        return list(self._secrets.keys())

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __getitem__(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise KeyError(f"Secret '{name}' not found")
        return value

    def __repr__(self) -> str:
        secrets_list = self.list_available()
        return f"KaggleSecrets(is_kaggle={self._is_kaggle}, secrets={secrets_list})"


def auto_load_secrets(
    set_env: bool = True,
    secrets_to_load: Optional[List[str]] = None
) -> Dict[str, Optional[str]]:
    """
    Load all known secrets and return as dict.

    This is the main convenience function for loading secrets.

    Args:
        set_env: Also set secrets in environment variables
        secrets_to_load: Specific secrets to load (default: all known)

    Returns:
        Dict mapping secret names to values (None if not set)

    Example:
        >>> secrets = auto_load_secrets()
        >>> if secrets.get("HF_TOKEN"):
        ...     print("HuggingFace ready!")
    """
    ks = KaggleSecrets()

    names = secrets_to_load or KaggleSecrets.KNOWN_SECRETS
    result = {}

    for name in names:
        value = ks.get(name)
        result[name] = value

        if set_env and value:
            os.environ[name] = value

    return result


def setup_huggingface_auth() -> bool:
    """
    Set up HuggingFace authentication from secrets.

    Looks for HF_TOKEN or HUGGING_FACE_HUB_TOKEN and configures
    the huggingface_hub library.

    Returns:
        True if authentication was set up successfully
    """
    secrets = KaggleSecrets()

    # Try both common token names
    token = secrets.get("HF_TOKEN") or secrets.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        return False

    # Set environment variables
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    # Try to login with huggingface_hub
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except ImportError:
        # huggingface_hub not installed, env vars are set
        return True
    except Exception:
        # Login failed but env vars are set
        return True


def setup_graphistry_auth() -> bool:
    """
    Set up Graphistry authentication from secrets.

    Looks for Graphistry credentials and registers with pygraphistry.

    Returns:
        True if authentication was set up successfully
    """
    secrets = KaggleSecrets()

    key_id = secrets.get("Graphistry_Personal_Key_ID")
    key_secret = secrets.get("Graphistry_Personal_Secret_Key")

    if not key_id or not key_secret:
        # Try alternative names
        username = secrets.get("GRAPHISTRY_USERNAME")
        password = secrets.get("GRAPHISTRY_PASSWORD")
        api_key = secrets.get("GRAPHISTRY_API_KEY")

        if api_key:
            try:
                import graphistry
                graphistry.register(api=3, token=api_key)
                return True
            except Exception:
                return False
        elif username and password:
            try:
                import graphistry
                graphistry.register(
                    api=3,
                    protocol="https",
                    server="hub.graphistry.com",
                    username=username,
                    password=password
                )
                return True
            except Exception:
                return False
        return False

    try:
        import graphistry
        graphistry.register(
            api=3,
            protocol="https",
            server="hub.graphistry.com",
            personal_key_id=key_id,
            personal_key_secret=key_secret
        )
        return True
    except ImportError:
        return False
    except Exception:
        return False


__all__ = [
    "KaggleSecrets",
    "auto_load_secrets",
    "setup_huggingface_auth",
    "setup_graphistry_auth",
]
