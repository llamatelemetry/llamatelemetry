"""
llamatelemetry.chat - Chat and Conversation Management

This module provides OpenAI-compatible chat completion support with:
- Multi-turn conversation management
- System prompts and role-based messages
- Chat history persistence
- Token counting and context management
- Streaming chat completions

Examples:
    Basic chat completion:
    >>> from llamatelemetry.chat import ChatEngine
    >>> chat = ChatEngine(engine)
    >>> chat.add_message("user", "What is quantum computing?")
    >>> response = chat.complete()
    >>> print(response)

    Streaming chat:
    >>> for chunk in chat.complete_stream():
    ...     print(chunk, end='', flush=True)
"""

from typing import List, Dict, Optional, Any, Iterator
import json
import time
import requests
from pathlib import Path


class Message:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str, name: Optional[str] = None):
        """
        Create a message.

        Args:
            role: Message role (system, user, assistant)
            content: Message content
            name: Optional name for the message sender
        """
        self.role = role
        self.content = content
        self.name = name
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg

    def __repr__(self) -> str:
        return f"Message(role='{self.role}', content='{self.content[:50]}...')"


class ChatEngine:
    """
    Manages chat conversations with history and context.

    Provides OpenAI-compatible chat completion interface with automatic
    conversation history management, token counting, and context window handling.

    Examples:
        >>> from llamatelemetry import InferenceEngine
        >>> from llamatelemetry.chat import ChatEngine
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True)
        >>> chat = ChatEngine(engine)
        >>> chat.add_system_message("You are a helpful AI assistant.")
        >>> chat.add_user_message("Explain photosynthesis")
        >>> response = chat.complete()
        >>> print(response)
    """

    def __init__(
        self,
        engine,
        system_prompt: Optional[str] = None,
        max_history: int = 20,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Initialize chat engine.

        Args:
            engine: InferenceEngine instance
            system_prompt: Optional system prompt
            max_history: Maximum messages to keep in history
            max_tokens: Default max tokens for completions
            temperature: Default temperature
        """
        self.engine = engine
        self.messages: List[Message] = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.temperature = temperature

        if system_prompt:
            self.add_system_message(system_prompt)

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> 'ChatEngine':
        """
        Add a message to the conversation.

        Args:
            role: Message role (system, user, assistant)
            content: Message content
            name: Optional sender name

        Returns:
            Self for chaining
        """
        msg = Message(role, content, name)
        self.messages.append(msg)

        # Trim history if needed (keep system messages)
        if len(self.messages) > self.max_history:
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            self.messages = system_msgs + other_msgs[-(self.max_history - len(system_msgs)):]

        return self

    def add_system_message(self, content: str) -> 'ChatEngine':
        """Add a system message."""
        return self.add_message("system", content)

    def add_user_message(self, content: str) -> 'ChatEngine':
        """Add a user message."""
        return self.add_message("user", content)

    def add_assistant_message(self, content: str) -> 'ChatEngine':
        """Add an assistant message."""
        return self.add_message("assistant", content)

    def complete(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Assistant's response text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Try OpenAI-compatible endpoint first
        try:
            response = requests.post(
                f"{self.engine.server_url}/v1/chat/completions",
                json={
                    "messages": [m.to_dict() for m in self.messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    **kwargs
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                assistant_message = data["choices"][0]["message"]["content"]
                self.add_assistant_message(assistant_message)
                return assistant_message

        except Exception:
            pass

        # Fallback to prompt-based completion
        prompt = self._build_prompt()
        result = self.engine.infer(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if result.success:
            self.add_assistant_message(result.text)
            return result.text
        else:
            return f"Error: {result.error_message}"

    def complete_stream(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate streaming chat completion.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        full_response = ""

        # Try OpenAI-compatible streaming endpoint
        try:
            response = requests.post(
                f"{self.engine.server_url}/v1/chat/completions",
                json={
                    "messages": [m.to_dict() for m in self.messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    **kwargs
                },
                stream=True,
                timeout=300
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            if line_str.strip() == 'data: [DONE]':
                                break

                            try:
                                data = json.loads(line_str[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        yield content
                            except json.JSONDecodeError:
                                pass

                self.add_assistant_message(full_response)
                return

        except Exception:
            pass

        # Fallback to non-streaming
        response = self.complete(max_tokens=max_tokens, temperature=temperature, **kwargs)
        yield response

    def _build_prompt(self) -> str:
        """Build a prompt from message history."""
        prompt_parts = []

        for msg in self.messages:
            role_name = msg.role.title()
            if msg.name:
                role_name = msg.name
            prompt_parts.append(f"{role_name}: {msg.content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def clear_history(self, keep_system: bool = True) -> 'ChatEngine':
        """
        Clear conversation history.

        Args:
            keep_system: Keep system messages

        Returns:
            Self for chaining
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []
        return self

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dictionaries."""
        return [m.to_dict() for m in self.messages]

    def save_history(self, filepath: str):
        """Save conversation history to JSON file."""
        data = {
            "messages": self.get_history(),
            "metadata": {
                "max_history": self.max_history,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timestamp": time.time()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_history(self, filepath: str) -> 'ChatEngine':
        """
        Load conversation history from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Self for chaining
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.messages = [
            Message(m['role'], m['content'], m.get('name'))
            for m in data['messages']
        ]

        if 'metadata' in data:
            meta = data['metadata']
            self.max_history = meta.get('max_history', self.max_history)
            self.max_tokens = meta.get('max_tokens', self.max_tokens)
            self.temperature = meta.get('temperature', self.temperature)

        return self

    def count_tokens(self) -> int:
        """
        Estimate token count for current conversation.

        Returns:
            Estimated token count
        """
        # Try using tokenize endpoint
        try:
            full_text = self._build_prompt()
            response = requests.post(
                f"{self.engine.server_url}/tokenize",
                json={"content": full_text},
                timeout=10
            )

            if response.status_code == 200:
                tokens = response.json().get('tokens', [])
                return len(tokens)
        except Exception:
            pass

        # Fallback to rough estimate (1 token ≈ 4 characters)
        full_text = self._build_prompt()
        return len(full_text) // 4

    def __repr__(self) -> str:
        return f"ChatEngine(messages={len(self.messages)}, tokens≈{self.count_tokens()})"


class ConversationManager:
    """
    Manages multiple conversation sessions.

    Allows switching between different conversation contexts,
    useful for handling multiple topics or users.

    Examples:
        >>> manager = ConversationManager(engine)
        >>> manager.create_conversation("coding", "You are a coding assistant")
        >>> manager.create_conversation("writing", "You are a writing coach")
        >>> manager.switch_to("coding")
        >>> manager.chat("How do I write a Python function?")
    """

    def __init__(self, engine):
        """
        Initialize conversation manager.

        Args:
            engine: InferenceEngine instance
        """
        self.engine = engine
        self.conversations: Dict[str, ChatEngine] = {}
        self.current_conversation: Optional[str] = None

    def create_conversation(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ChatEngine:
        """
        Create a new conversation.

        Args:
            name: Conversation name
            system_prompt: Optional system prompt
            **kwargs: Additional ChatEngine parameters

        Returns:
            Created ChatEngine instance
        """
        chat = ChatEngine(self.engine, system_prompt=system_prompt, **kwargs)
        self.conversations[name] = chat

        if self.current_conversation is None:
            self.current_conversation = name

        return chat

    def switch_to(self, name: str) -> ChatEngine:
        """
        Switch to a conversation.

        Args:
            name: Conversation name

        Returns:
            ChatEngine instance

        Raises:
            KeyError: If conversation doesn't exist
        """
        if name not in self.conversations:
            raise KeyError(f"Conversation '{name}' not found")

        self.current_conversation = name
        return self.conversations[name]

    def get_current(self) -> ChatEngine:
        """Get current conversation."""
        if self.current_conversation is None:
            raise RuntimeError("No active conversation")
        return self.conversations[self.current_conversation]

    def chat(self, message: str, **kwargs) -> str:
        """
        Send message to current conversation.

        Args:
            message: User message
            **kwargs: Completion parameters

        Returns:
            Assistant response
        """
        chat = self.get_current()
        chat.add_user_message(message)
        return chat.complete(**kwargs)

    def list_conversations(self) -> List[str]:
        """List all conversation names."""
        return list(self.conversations.keys())

    def delete_conversation(self, name: str):
        """Delete a conversation."""
        if name in self.conversations:
            del self.conversations[name]

            if self.current_conversation == name:
                self.current_conversation = next(iter(self.conversations), None)

    def save_all(self, directory: str):
        """
        Save all conversations to directory.

        Args:
            directory: Directory to save conversations
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for name, chat in self.conversations.items():
            filepath = dir_path / f"{name}.json"
            chat.save_history(str(filepath))

    def load_all(self, directory: str):
        """
        Load all conversations from directory.

        Args:
            directory: Directory containing conversation files
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return

        for filepath in dir_path.glob("*.json"):
            name = filepath.stem
            chat = ChatEngine(self.engine)
            chat.load_history(str(filepath))
            self.conversations[name] = chat

            if self.current_conversation is None:
                self.current_conversation = name
