import os
import json
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()


class SentrySolChat:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("LLM_MODEL", "mistral-large-latest")
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

    def get_default_system_prompt(self) -> str:
        """Default system prompt for SentrySol chat"""
        return """You are SentrySol, an advanced Solana blockchain security analyst and expert. You specialize in:

- Analyzing Solana wallet addresses, transactions, and token transfers
- Identifying security risks, suspicious activities, and potential threats
- Providing detailed threat assessments with risk scores
- Explaining blockchain transaction patterns and behaviors
- Offering security recommendations and best practices

Your responses should be:
- Accurate and technically precise
- Security-focused with actionable insights  
- Easy to understand for both technical and non-technical users
- Structured with clear threat levels and recommendations

Always prioritize security analysis and provide concrete evidence for your assessments."""

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        conversation_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Send chat message to Mistral AI with dynamic prompts

        Args:
            user_message: The user's question/message
            system_prompt: Custom system prompt (uses default if None)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            conversation_history: Previous messages for context

        Returns:
            Dict containing response, metadata, and usage info
        """

        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Make API request
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # Extract response content
            assistant_message = data["choices"][0]["message"]["content"]

            return {
                "success": True,
                "response": assistant_message,
                "model": data.get("model", self.model),
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                },
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt_length": len(system_prompt),
                    "user_message_length": len(user_message),
                },
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "error_type": "request_error",
            }

        except KeyError as e:
            return {
                "success": False,
                "error": f"Unexpected response format: {str(e)}",
                "error_type": "response_format_error",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unknown_error",
            }

    def chat_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Stream chat response from Mistral AI

        Args:
            user_message: The user's question/message
            system_prompt: Custom system prompt (uses default if None)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length

        Yields:
            Streaming response chunks
        """

        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload, stream=True, timeout=60
            )

            response.raise_for_status()

            # Stream the response
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"Error: {str(e)}"


# Initialize global instance
sentrysol_chat = SentrySolChat()


def get_chat_instance() -> SentrySolChat:
    """Get the global chat instance"""
    return sentrysol_chat
