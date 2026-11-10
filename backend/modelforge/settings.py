from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MF_")
    database_url: str = "sqlite:///./modelforge.db"
    redis_url: str = "redis://localhost:6379/0"
    artifact_root: Path = Path("./storage")
    tracking_uri: str = "sqlite:///./mlflow.db"
    eager: bool = False
    upload_limit: int = 10 * 1024 * 1024


settings = Settings()
