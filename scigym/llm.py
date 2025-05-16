from abc import ABC, abstractmethod
from typing import Dict, List

import anthropic
from google import genai
from google.genai import types
from openai import OpenAI


class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    Each LLM provider should implement this interface.
    """

    @abstractmethod
    def initialize(self, api_key: str, model_name: str, system_prompt: str, max_length: int):
        """Initialize the model provider with necessary parameters"""

    @abstractmethod
    def get_response(self, user_message: str) -> tuple:
        """Get a response from the model and return (response_text, usage_stats)"""

    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full message history in a standard format"""


class AnthropicProvider(ModelProvider):
    """Implementation for Anthropic Claude models"""

    def initialize(
        self, api_key: str, model_name: str, system_prompt: str, max_length: int, temperature: float
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.messages = []
        self.temperature = temperature

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full message history"""
        return self.messages

    def get_response(self, user_message: str) -> tuple:
        """Get a response from Claude API"""
        # Add user message to history
        self.add_message("user", user_message)

        # Call Claude API
        response = self.client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=self.messages,
            max_tokens=self.max_length,
            temperature=self.temperature,
        )

        # Get response text
        assert len(response.content) > 0
        assert isinstance(response.content[0], anthropic.types.TextBlock)
        response_text = response.content[0].text

        # Add assistant response to history
        self.add_message("assistant", response_text)

        # Return usage statistics
        usage_stats = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return response_text, usage_stats


class OpenAIProvider(ModelProvider):
    """Implementation for OpenAI GPT models"""

    def initialize(
        self, api_key: str, model_name: str, system_prompt: str, max_length: int, temperature: float
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature

        # Initialize with system message
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full message history"""
        return self.messages

    def get_response(self, user_message: str) -> tuple:
        """Get a response from OpenAI API"""
        # Add user message to history
        self.add_message("user", user_message)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=self.max_length,
            temperature=self.temperature,
        )

        # Get response text
        response_text = response.choices[0].message.content

        # Add assistant response to history
        self.add_message("assistant", response_text)

        # Return usage statistics
        usage_stats = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return response_text, usage_stats


class GeminiProvider(ModelProvider):
    """Implementation for Google Gemini models"""

    def initialize(
        self, api_key: str, model_name: str, system_prompt: str, max_length: int, temperature: float
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature

        # Initialize Gemini chat
        chat_config = types.GenerateContentConfig(
            system_instruction=self.system_prompt, temperature=self.temperature
        )
        self.chat = self.client.chats.create(model=self.model_name, config=chat_config)

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.
        Note: For Gemini, we don't need to manually track messages since the chat object does it.
        This method is a no-op for consistency with the interface.
        """

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get the full message history.
        For Gemini, we convert from their format to our standard format.
        """
        gemini_history = self.chat.get_history()
        standard_messages = []

        for i, msg in enumerate(gemini_history):
            role = "user" if i % 2 == 0 else "assistant"
            content = msg.parts[0].text if hasattr(msg, "parts") else str(msg)
            standard_messages.append({"role": role, "content": content})

        return standard_messages

    def get_response(self, user_message: str) -> tuple:
        """Get a response from Gemini API"""
        # Send message to Gemini chat
        response = self.chat.send_message(user_message)
        response_text = response.text

        # Return usage statistics
        usage_stats = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }
        if (
            hasattr(response.usage_metadata, "thoughts_token_count")
            and response.usage_metadata.thoughts_token_count
        ):
            usage_stats["thoughts_tokens"] = response.usage_metadata.thoughts_token_count

        return response_text, usage_stats


class LLM:
    """
    Unified interface for large language models in chat mode.
    Supports multiple model providers through a plugin architecture.
    """

    # Registry of supported model providers
    _providers = {"claude": AnthropicProvider, "gpt": OpenAIProvider, "gemini": GeminiProvider}

    @classmethod
    def register_provider(cls, prefix: str, provider_class):
        """
        Register a new model provider

        Args:
            prefix: String prefix to identify models from this provider
            provider_class: Class implementing the ModelProvider interface
        """
        cls._providers[prefix] = provider_class

    def __init__(
        self, model_name: str, api_key: str, system_prompt: str, temperature: float, max_length=8192
    ):
        """
        Initialize with a model name and API key.

        Args:
            model_name: Name of the LLM to use (e.g., "claude-3-opus-20240229", "gpt-4", "gemini-pro")
            api_key: API key for the LLM service
            system_prompt: Initial system prompt for the conversation
            log_file: File to log conversations
            max_length: Maximum response length
        """
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.temperature = temperature

        # Open log file

        # Token usage tracking
        self.input_total_tokens = 0
        self.output_total_tokens = 0

        # Initialize the right provider
        self._initialize_provider()

    def _initialize_provider(self):
        """Determine and initialize the appropriate provider for the model"""
        provider_key = None

        # Find matching provider by prefix
        for prefix, provider_class in self._providers.items():
            if prefix in self.model_name.lower():
                provider_key = prefix
                self.provider = provider_class()
                break

        if not self.provider:
            raise ValueError(
                f"Unsupported model: {self.model_name}. Please register a provider for this model type."
            )

        # Initialize the provider with our settings
        self.provider.initialize(
            api_key=self.api_key,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
        )

    def get_message(self) -> dict:
        """
        Get the chat history organized by iteration ID.

        Returns:
            dict: A nested dictionary with iteration_id as the outer key and user/assistant messages as inner keys
        """
        messages = self.provider.get_messages()
        chat_history = {}
        iteration_id = 0

        # Skip system message if present
        start_idx = 1 if messages and messages[0]["role"] == "system" else 0

        # Process messages in pairs (user, assistant)
        for i in range(start_idx, len(messages), 2):
            if i + 1 < len(messages):
                # Complete pair of user and assistant messages
                chat_history[f"Iteration {iteration_id}"] = {
                    "user": messages[i]["content"],
                    "assistant": messages[i + 1]["content"],
                }
                iteration_id += 1
            else:
                # Last user message without assistant response
                chat_history[f"Iteration {iteration_id}"] = {
                    "user": messages[i]["content"],
                    "assistant": None,
                }

        return chat_history

    def return_response(self, user_message: str | list) -> str:
        """
        Get a response from the LLM based on a user message and conversation history.

        Args:
            user_message: The message to send to the LLM

        Returns:
            LLM's response as a string
        """
        # Convert list to string if needed
        if isinstance(user_message, list):
            user_message = str(user_message)

        # Get response from provider
        response_text, usage_stats = self.provider.get_response(user_message)

        # Update token usage
        if "input_tokens" in usage_stats and usage_stats["input_tokens"]:
            self.input_total_tokens += usage_stats["input_tokens"]
        if "output_tokens" in usage_stats and usage_stats["output_tokens"]:
            self.output_total_tokens += usage_stats["output_tokens"]
        if "thoughts_tokens" in usage_stats and usage_stats["thoughts_tokens"]:
            self.output_total_tokens += usage_stats["thoughts_tokens"]

        # Log the conversation
        # print(f"[ENV_INPUT]\n\n{user_message}\n\n[LLM_OUTPUT]\n\n{response_text}\n")

        return response_text


