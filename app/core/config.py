from pydantic_settings import BaseSettings, SettingsConfigDict
 
class Settings(BaseSettings):
    DASHSCOPE_API_KEY: str
    DASHSCOPE_API_URL: str
    ENV: str = "dev"
    MODEL_NAME: str = "qwen3.5-plus"
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # 读取.env文件
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()