from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.llm.base import LLMFactory
from src.models.entities import LLMConfig
from src.memory.base import ConversationMemory


