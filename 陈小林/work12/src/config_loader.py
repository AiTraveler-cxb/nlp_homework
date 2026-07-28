from pathlib import Path
from src.models.entities import RedisConfig
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_DIR= ROOT / 'config'
def load_config(file_name:str = 'local.yaml'):

    with open(CONFIG_DIR / file_name, 'r') as f:
        conf_data = yaml.safe_load(f)

    redis_conf = RedisConfig(**conf_data['redis'])
    return redis_conf