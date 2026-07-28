from __future__ import annotations
import threading

from src.models.entities import RedisConfig
from redis import ConnectionPool, Redis

class RedisPoolManager:

    _instance: RedisPoolManager | None = None
    _lock: threading.Lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.pools =  {}
                    cls._instance.clients = {}
        return cls._instance

    def _get_pool_key(self, conf: RedisConfig):
        return f'{conf.host}:{conf.port}:{conf.db}'

    def _get_pool(self, conf: RedisConfig):
        """内部方法：生成连接池键"""
        key = self._get_pool_key(conf)
        if key not in self.pools:
            self.pools[key] = ConnectionPool(
                host=conf.host,
                port=conf.port,
                password=conf.password,
                db=conf.db,
                max_connections=conf.max_connections,
                socket_timeout=conf.socket_timeout,
                socket_connect_timeout=conf.socket_connect_timeout,
                retry_on_timeout=conf.retry_on_timeout,
                health_check_interval=conf.health_check_interval,
            )
        return self.pools[key]

    def get_client(self, conf: RedisConfig):
        key = self._get_pool_key(conf)
        if key in self.clients:
            return self.clients[key]
        with self._lock:
            if key not in self.clients:
                self.clients[key] = Redis(connection_pool=self._get_pool(conf))
        return self.clients[key]

    def close(self, conf: RedisConfig):
       key = self._get_pool_key(conf)
       if key in self.pools:
           self.pools[key].disconnect()
           del self.pools[key]
       if key in self.clients:
           del self.clients[key]

    def close_all(self):
        for key in self.pools:
            self.pools[key].disconnect()
        self.pools.clear()
        self.clients.clear()