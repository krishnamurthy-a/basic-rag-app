import yaml
import threading

class Configs:
    _CONFIG_FILE = "resources/configs.yaml"
    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.config: dict = None
        with open(Configs._CONFIG_FILE, "r") as f:
            self.config = yaml.safe_load(f)

    @classmethod
    def get_configs(cls) -> dict:
        if not cls._instance:
            cls._instance = Configs()
        return cls._instance.config

def load_configs():
    return Configs.get_configs()
