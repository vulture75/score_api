from dataclasses import dataclass
from os import path, environ
from dotenv import dotenv_values

base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

BASE_DIR = base_dir

DB_POOL_RECYCLE: int = 900
DB_ECHO: bool = True

ENV_FILE = dotenv_values(".env")


@dataclass
class Config:
    """
    기본 Configuration
    """

    @staticmethod
    def get_env_value(key: str):
        global ENV_FILE
        try:
            return ENV_FILE[key]
        except KeyError as ex:
            print(ex)
            return None

    @staticmethod
    def make_env_value_key(key_name: str, service: str = None, stage: str = None):
        def get_stage_text(stage: str):
            match (stage):
                case "production":
                    return "prod"
                case "development":
                    return "test"
                case _:
                    return stage

        key_text = key_name
        if service:
            key_text = f"{service}_{key_text}"
        if stage:
            stage_text = get_stage_text(stage)
            key_text += f"_{stage_text}"

        key_text = key_text.upper()
        print(f"key_text: {key_text}")
        return key_text


@dataclass
class LocalConfig(Config):
    PROJ_RELOAD: bool = True


@dataclass
class ProdConfig(Config):
    PROJ_RELOAD: bool = False


def conf():
    """
    환경 불러오기
    :return:
    """
    config = dict(prod=ProdConfig(), local=LocalConfig())
    return config.get(environ.get("API_ENV", "local"))
