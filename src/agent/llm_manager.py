import os
from src.utils.logger import setup_logger

logger = setup_logger("LLMManager")

class LLMManager:
    def __init__(self, llm_type: str = "auto"):
        self.llm = None
        self.llm_type = None
        self.failed_providers = []
        self._init_llm(llm_type)
    
    def _init_llm(self, llm_type: str):
        # 初始化LLM
        if llm_type == "auto":
            for provider in ["openai", "qwen", "deepseek"]:
                if self._try_init(provider):
                    return
        else:
            self._try_init(llm_type)
        
        # 所有LLM都失败了
        if not self.llm:
            error_msg = "所有LLM初始化失败，请检查API key配置"
            if self.failed_providers:
                error_msg += f"\n失败详情: {'; '.join(self.failed_providers)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _try_init(self, provider: str) -> bool:
        # 尝试初始化指定LLM
        try:
            if provider == "openai" and os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    max_completion_tokens=int(os.getenv("OPENAI_TOKEN", "5120")),
                    temperature=0.7,
                    timeout=60,
                    max_retries=3,
                )
                self.llm_type = "OpenAI"
                
            elif provider == "qwen" and os.getenv("DASHSCOPE_API_KEY"):
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("DASHSCOPE_MODEL", "qwen-turbo"),
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url=os.getenv("DASHSCOPE_BASE_URL"),
                    max_completion_tokens=int(os.getenv("DASHSCOPE_TOKEN", "5120")),
                    temperature=0.7,
                    timeout=60,
                    max_retries=3,
                )
                self.llm_type = "Qwen"
                
            elif provider == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                    max_completion_tokens=int(os.getenv("DEEPSEEK_TOKEN", "5120")),
                    temperature=0.7,
                    timeout=60,
                    max_retries=3,
                )
                self.llm_type = "DeepSeek"
            
            if self.llm:
                logger.info(f"{self.llm_type} 初始化成功")
                return True
                
        except Exception as e:
            self.failed_providers.append(f"{provider}: {str(e)}")
            logger.warning(f"{provider} 初始化失败，尝试下一个")
        
        return False
