from pydantic import BaseModel

class LLMConfig(BaseModel):
    model_type: str = "openai"
    model: str = "gpt-3.5-turbo"
    base_url:str =  ""
    api_key: str = 'API_TOKEN'
    temperature: float = 0.7

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None
    max_connections: int = 10
    timeout: int = 300
    retry_on_timeout: int = 300
    socket_timeout: int = 300
    socket_connect_timeout: int = 300
    health_check_interval:int = 30

