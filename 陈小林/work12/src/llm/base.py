from abc import ABC, abstractmethod
from typing import Dict

from langchain_openai import ChatOpenAI
from src.models.entities import LLMConfig

class LLMProvider(ABC):

    @abstractmethod
    def create_llm(self, llm_config: LLMConfig):
        pass

class DeepSeekProvider(LLMProvider):

    def create_llm(self, llm_config: LLMConfig):
        if not llm_config.api_key:
            raise ValueError("API key required")

        return ChatOpenAI(
            model=llm_config.model or "deepseek-chat",
            temperature=llm_config.temperature or 0.7,
            base_url=llm_config.base_url or "https://api.deepseek.com/v1",
            api_key=llm_config.api_key
        )

class LLMFactory:
    _llm_providers: Dict[str, LLMProvider] = {}

    @classmethod
    def register_provider(self, model_type: str, provider: LLMProvider):
        self._llm_providers[model_type] = provider

    @classmethod
    def create_llm(self, llm_config: LLMConfig):
        provider = self._llm_providers.get(llm_config.model_type)
        if not provider:
            raise ValueError(f"Unknown model type: {llm_config.model_type}")

        return provider.create_llm(llm_config)

# 注册提供者
LLMFactory.register_provider("deepseek", DeepSeekProvider())