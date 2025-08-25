import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_cerebras import ChatCerebras  # Tambahkan import Cerebras

load_dotenv()


class SentrySolChat:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("LLM_MODEL", "SentrySol-Standart")
        self.cerebras_api_key = os.getenv("OPENAI_API_KEY")
        self.cerebras_model = os.getenv("CEREBRAS_MODEL", "SentrySol-Premium")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=self.api_key)
        # Tidak perlu inisialisasi Cerebras di sini, karena bisa gagal jika API key tidak ada

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

        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history:
                if (
                    isinstance(msg, dict)
                    and "role" in msg
                    and "content" in msg
                    and isinstance(msg["role"], str)
                    and isinstance(msg["content"], str)
                ):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        # Utamakan Cerebras, fallback ke Mistral jika gagal
        try:
            if not self.cerebras_api_key:
                raise Exception("API_KEY not set")
            cerebras_client = ChatCerebras(
                model=self.cerebras_model,
                openai_api_key=self.cerebras_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Gunakan format prompt yang sama seperti Mistral
            prompt = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
            )
            response = cerebras_client.invoke(prompt)
            assistant_message = (
                response.content if hasattr(response, "content") else str(response)
            )
            return {
                "success": True,
                "response": assistant_message,
                "model": self.cerebras_model,
                "usage": {},
                "metadata": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt_length": len(system_prompt),
                    "user_message_length": len(user_message),
                    "engine": "SentrySol",
                },
            }
        except Exception as cerebras_exc:
            # Fallback ke Mistral
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                assistant_message = response.choices[0].message.content

                return {
                    "success": True,
                    "response": assistant_message,
                    "model": getattr(response, "model", self.model),
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", None
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    },
                    "metadata": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "system_prompt_length": len(system_prompt),
                        "user_message_length": len(user_message),
                        "engine": "SentrySol",
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"API request failed: {str(e)}",
                    "error_type": "request_error",
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

        # Utamakan Cerebras, fallback ke Mistral jika gagal
        try:
            if not self.cerebras_api_key:
                raise Exception("CEREBRAS_API_KEY not set")
            cerebras_client = ChatCerebras(
                model=self.cerebras_model,
                openai_api_key=self.cerebras_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prompt = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
            )
            response = cerebras_client.invoke(prompt)
            # Anggap response.content adalah string panjang, yield per kalimat/baris
            for chunk in response.content.splitlines():
                if chunk.strip():
                    yield chunk
        except Exception as cerebras_exc:
            # Fallback ke Mistral
            try:
                stream_response = self.client.chat.stream(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for chunk in stream_response:
                    delta = getattr(chunk.data.choices[0], "delta", None)
                    if delta and hasattr(delta, "content") and delta.content:
                        yield delta.content
            except Exception as e:
                yield f"Error: {str(e)}"


# Initialize global instance
sentrysol_chat = SentrySolChat()


def get_chat_instance() -> SentrySolChat:
    """Get the global chat instance"""
    return sentrysol_chat
