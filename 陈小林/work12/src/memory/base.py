import json
import threading
from queue import Queue
from typing import Optional, ClassVar

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, PrivateAttr
from redis import Redis
from src.models.entities import RedisConfig
from src.services.cache import RedisPoolManager


class ConversationMemory(BaseModel):
    session_id: str
    messages: list[tuple[str, str]] = Field(default_factory=list)
    max_length: int = Field(default=10)
    _prompt_template: ClassVar[Optional[ChatPromptTemplate]] = None

    @classmethod
    def _get_prompt_template(cls):
        if cls._prompt_template is not None:
            return cls._prompt_template
        cls._prompt_template =  ChatPromptTemplate.from_messages([
            ('system', '你是一个有用的AI助手'),
            MessagesPlaceholder(variable_name="history"),
            ('human', '{input}')
        ])
        return cls._prompt_template

    def add_message(self, user_message:str, assistant_message:str):
        if len(self.messages) >= self.max_length:
            print(f'存储记忆超过最大值: {self.max_length}, 删除最老的记录:{self.messages.pop(0)}')
        self.messages.append((user_message, assistant_message))

    def get_context(self, input_str:str):
        history_msg = []
        for user_message, assistant_message in self.messages:
            history_msg.append(HumanMessage(content=user_message))
            history_msg.append(AIMessage(content=assistant_message))
        self._get_prompt_template()
        return self._prompt_template.format_prompt(input=input_str, history=history_msg)


class RedisMemory(ConversationMemory):
    conf: RedisConfig
    _key_prefix: str = "session:"
    _client: Redis = PrivateAttr()
    _sync_queue:Queue = PrivateAttr()
    def model_post_init(self, __context):
        self._client = RedisPoolManager().get_client(self.conf)
        self._sync_queue = Queue()
        self._start_sync_task()
        self._recover_from_redis()

    @property
    def _redis_key(self) -> str:
        """生成 Redis key"""
        return f"{self._key_prefix}{self.session_id}:history"

    def _recover_from_redis(self):
        """从 Redis 中恢复数据"""
        if not self._client:
            return
        messages = self._client.lrange(self._redis_key, 0, -1)
        for message in messages:
            message = json.loads(message)
            self.add_message(message["user"], message["assistant"])

    def _start_sync_task(self):
        def sync_task():
            while True:
                try:
                    user_message, assistant_message  = self._sync_queue.get()
                    self._save_to_redis(user_message, assistant_message )
                except Exception as e:
                    print(f"同步任务出错: {e}")

        threading.Thread(target=sync_task, daemon=True).start()

    def add_sync_queue(self, user_message:str, assistant_message:str):
        """添加同步任务"""
        self._sync_queue.put((user_message, assistant_message))

    def add_message(self, user_message:str, assistant_message:str):
        super().add_message(user_message, assistant_message)
        self.add_sync_queue(user_message, assistant_message)

    def _save_to_redis(self, user_message:str, assistant_message:str):
        """保存单条记录到 Redis"""
        if not self._client:
            return
        message = json.dumps({"user": user_message, "assistant": assistant_message})

        pipe = self._client.pipeline()
        pipe.rpush(self._redis_key, message)
        pipe.ltrim(self._redis_key, -self.max_length, -1)
        pipe.execute()
