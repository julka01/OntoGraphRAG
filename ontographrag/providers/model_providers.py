import os
import json
import httpx
from abc import ABC, abstractmethod
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
from huggingface_hub import InferenceClient
from typing import Dict, Any

from langchain_core.runnables.base import Runnable
from ontographrag.kg.utils.common_functions import _resolve_huggingface_embedding_model

class ModelProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        pass

class LangChainRunnableAdapter(Runnable):
    def __init__(self, provider: ModelProvider, model: str, temperature: float = 1.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def invoke(self, input, config=None) -> str:
        # Handle ChatPromptTemplate inputs (system and user messages)
        if isinstance(input, dict):
            # Direct dict input - extract model name if provided
            messages = input.get("messages", [])
            model_name = input.get("model", self.model)
        elif hasattr(input, "to_messages"):
            # ChatPromptValue object
            messages = input.to_messages()
            model_name = self.model
        else:
            # Fallback
            messages = []
            model_name = self.model

        system_prompt = ""
        user_prompt = ""

        for message in messages:
            if hasattr(message, "type") and hasattr(message, "content"):
                if message.type == "system":
                    system_prompt = message.content
                elif message.type == "human":
                    user_prompt = message.content

        # If no explicit messages found, try simple string input
        if not system_prompt and not user_prompt:
            if isinstance(input, dict):
                user_prompt = input.get("input", input.get("text", input.get("chunk_text", "")))
            elif hasattr(input, "content"):
                user_prompt = input.content
            elif isinstance(input, str):
                user_prompt = input
            else:
                user_prompt = str(input)

        return self.provider.generate(
            system_prompt, user_prompt, model_name, temperature=self.temperature
        )

    def __class_getitem__(cls, item):
        return cls
    
    def with_structured_output(self, schema, **kwargs):
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self
    
    def bind(self, **kwargs):
        """Add bind method for LangChain compatibility"""
        return self

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

class OpenAIProvider(ModelProvider):
    def __init__(self):
        self.client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=LLM_TIMEOUT_SECONDS,
        )

    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        kwargs.setdefault("temperature", 1.0)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        return response.choices[0].message.content

class OllamaProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        import ollama
        try:
            # Check if JSON format is explicitly requested in kwargs
            force_json = kwargs.pop('force_json', False)
            
            # Build the chat request
            chat_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {"num_ctx": 16384},
            }
            
            # Only enforce JSON format if explicitly requested
            if force_json:
                chat_kwargs['format'] = 'json'
            
            # Add any remaining kwargs
            chat_kwargs.update(kwargs)
            
            response = ollama.chat(**chat_kwargs)
            content = response['message']['content']
            
            # If JSON was requested, validate the response
            if force_json:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON response from model: {content}")
            
            return content
        except Exception as e:
            if "not found" in str(e).lower():
                return f"Error: Model '{model}' not found. Please run 'ollama pull {model}' to download it."
            else:
                return f"Ollama error: {str(e)}"

class GeminiProvider(ModelProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        kwargs.pop("temperature", None)  # pass via generation_config instead
        gemini_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(temperature=0)
        )
        response = gemini_model.generate_content(
            contents=[system_prompt, user_prompt],
            **kwargs
        )
        return response.text

class HuggingFaceProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        kwargs.setdefault("temperature", 0.0)
        response = client.text_generation(
            prompt=full_prompt,
            model=model,
            **kwargs
        )
        return response

class DeepSeekProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}
        kwargs.setdefault("temperature", 0)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        }
        
        response = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class OpenRouterProvider(ModelProvider):
    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
            "X-Title": "KG RAG Tool"
        }
        kwargs.setdefault("temperature", 0)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        }
        
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def with_structured_output(self, schema, **kwargs):
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self
    
    def bind(self, **kwargs):
        """Add bind method for LangChain compatibility"""
        return self


class TemperatureLockedProvider(ModelProvider):
    """Wrap a provider and force a fixed temperature on every generate call."""

    def __init__(self, provider: ModelProvider, temperature: float = 0.0):
        self.provider = provider
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str, model: str, **kwargs) -> str:
        kwargs = dict(kwargs)
        kwargs["temperature"] = self.temperature
        return self.provider.generate(system_prompt, user_prompt, model, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.provider, name)

def get_provider(provider: str, model: str = None, **kwargs) -> ModelProvider:
    providers: Dict[str, Any] = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider
    }
    
    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return provider_class()

# alias for compatibility
get_llm_provider = get_provider

def get_embedding_model(provider="huggingface"):
    """Get an embedding model from different providers

    Args:
        provider: Provider to use ('huggingface', 'openai', 'vertexai')

    Returns:
        LangChain embedding model
    """
    if provider == "huggingface":
        try:
            # Try to import from the correct package first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                except ImportError:
                    try:
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                    except ImportError:
                        print("❌ No compatible HuggingFace embeddings package found")
                        raise

            print("✓ Importing HuggingFaceEmbeddings...")
            model_name = _resolve_huggingface_embedding_model()
            embedder = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU by default for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ HuggingFaceEmbeddings initialized successfully")
            return embedder
        except Exception as e:
            print(f"❌ HuggingFace initialization failed: {e}")
            raise ImportError("huggingface embeddings not available. Install with: pip install sentence-transformers transformers torch")
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        )
    elif provider == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(
            model=os.getenv("VERTEXAI_EMBEDDING_MODEL", "text-embedding-005").strip()
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Choose from 'huggingface', 'openai', 'vertexai'")

def get_embedding_method(provider_name=None):
    """Get the configured embedding method from environment

    Args:
        provider_name: Optional provider override ('huggingface', 'openai', 'vertexai')

    Returns:
        tuple: (provider_name, embedding_model)
    """
    if provider_name is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # Default to huggingface
    else:
        provider = provider_name

    try:
        embedder = get_embedding_model(provider)
        return provider, embedder
    except Exception as e:
        print(f"Failed to initialize {provider} embeddings, falling back to OpenAI: {e}")
        # Fallback to OpenAI if HuggingFace fails
        try:
            embedder = get_embedding_model("openai")
            return "openai", embedder
        except Exception as e2:
            raise RuntimeError(f"Failed to initialize any embedding model. OpenAI error: {e2}")
