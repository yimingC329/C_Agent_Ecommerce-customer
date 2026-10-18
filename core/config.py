from pydantic_settings import BaseSettings
from enum import Enum
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"

class ServiceType(str, Enum):
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"

class Settings(BaseSettings):
    # Deepseek settings
    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str
    DEEPSEEK_MODEL: str
    
    # Vision Model settings (独立配置)
    VISION_API_KEY: str
    VISION_BASE_URL: str
    VISION_MODEL: str
    
    # Ollama settings
    OLLAMA_BASE_URL: str
    OLLAMA_CHAT_MODEL: str
    OLLAMA_REASON_MODEL: str
    OLLAMA_EMBEDDING_MODEL: str
    OLLAMA_AGENT_MODEL: str
    # Service selection
    CHAT_SERVICE: ServiceType = ServiceType.DEEPSEEK
    REASON_SERVICE: ServiceType = ServiceType.OLLAMA
    AGENT_SERVICE: ServiceType = ServiceType.DEEPSEEK

    # Intent routing / hard-negative mining settings
    # structured: 保持原有 JSON 结构化路由；logprob: 使用 DeepSeek FIM token logprobs
    INTENT_ROUTER_MODE: str = "boundary_v2"
    INTENT_FEWSHOT_ENABLED: bool = True
    INTENT_FEWSHOT_MAX_EXAMPLES: int = 15
    INTENT_CANONICAL_MAX_EXAMPLES: int = 9
    INTENT_CANONICAL_ANCHOR_PATH: str = str(
        ROOT_DIR / "data" / "intent" / "canonical_anchors.jsonl"
    )
    INTENT_CONFIDENCE_THRESHOLD: float = 0.8
    INTENT_HARD_NEGATIVE_MAX_RATIO: float = 0.03
    INTENT_BOUNDARY_BLEND_WEIGHT: float = 0.5
    INTENT_HARD_NEGATIVE_PATH: str = str(
        ROOT_DIR / "data" / "intent" / "hard_negatives.jsonl"
    )
    INTENT_AUDIT_ENABLED: bool = True
    INTENT_AUDIT_PATH: str = str(
        ROOT_DIR / "data" / "intent" / "intent_audit.jsonl"
    )
    HALLUCINATION_GUARD_ENABLED: bool = True
    HALLUCINATION_GUARD_MAX_RETRIES: int = 1
    HALLUCINATION_GUARD_MIN_SCORE: float = 0.8
    HALLUCINATION_AUDIT_ENABLED: bool = True
    HALLUCINATION_AUDIT_PATH: str = str(
        ROOT_DIR / "data" / "hallucination" / "grounding_audit.jsonl"
    )
    # 官方 DeepSeek 的 token logprobs 在 beta FIM Completion 接口上提供。
    # 如果使用第三方兼容端点，可通过下面两项单独覆盖。
    DEEPSEEK_LOGPROB_BASE_URL: str = ""
    INTENT_LOGPROB_MODEL: str = ""
    
    # Search settings
    SERPAPI_KEY: str
    SEARCH_RESULT_COUNT: int = 3
    
    # Database settings
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    
    # Neo4j settings
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # JWT settings
    SECRET_KEY: str = "your-secret-key"  # 在生产环境中使用安全的密钥
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_CACHE_EXPIRE: int = 3600
    REDIS_CACHE_THRESHOLD: float = 0.8
    
    # Embedding settings 
    EMBEDDING_TYPE: str = "ollama"  # ollama 或 sentence_transformer
    EMBEDDING_MODEL: str = "bge-m3"  # ollama embedding模型
    EMBEDDING_THRESHOLD: float = 0.90  # 语义相似度阈值
    
    # GraphRAG settings
    GRAPHRAG_PROJECT_DIR: str = "llm_backend/app/graphrag"  # GraphRAG项目目录
    GRAPHRAG_DATA_DIR: str = "data"                         # 数据目录名称
    GRAPHRAG_QUERY_TYPE: str = "local"                      # 查询类型
    GRAPHRAG_RESPONSE_TYPE: str = "text"                    # 响应类型
    GRAPHRAG_COMMUNITY_LEVEL: int = 3                       # 社区级别
    GRAPHRAG_DYNAMIC_COMMUNITY: bool = False                # 是否动态选择社区
    
    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def REDIS_URL(self) -> str:
        """构建Redis URL"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def NEO4J_CONN_URL(self) -> str:
        """构建Neo4j连接URL"""
        return f"{self.NEO4J_URL}"
    
    class Config:
        env_file = str(ENV_FILE)  # 使用绝对路径
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings() 
