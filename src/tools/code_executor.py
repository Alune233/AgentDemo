import os
import time
import requests
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class CodeExecutionResult:
    success: bool
    output: str
    error: str
    execution_time: float

class SandboxFusionExecutor:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # 从环境变量读取超时配置
        self.default_timeout = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))
        
        logger.info(f"初始化SandboxFusion执行器: {self.base_url}, 默认超时: {self.default_timeout}秒")
    
    def execute_code(self, code: str, language: str = "python", timeout: int = 30) -> CodeExecutionResult:
        # 使用环境变量的超时配置
        timeout = self.default_timeout
            
        logger.debug(f"执行{language}代码，超时{timeout}秒")
        start_time = time.time()
        
        try:
            payload = {"code": code, "language": language}
            response = self.session.post(
                f"{self.base_url}/run_code",
                json=payload, 
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('status') == 'Success'
                
                run_result = result.get('run_result', {})
                output = run_result.get('stdout', '')
                error = run_result.get('stderr', '')
                
                if not success:
                    message = result.get('message', '')
                    if message:
                        error = f"{error}\n{message}".strip()
                
                logger.info(f"代码执行{'成功' if success else '失败'}，耗时{execution_time:.3f}秒")
                
                return CodeExecutionResult(success, output, error, execution_time)
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"代码执行失败: {error_msg}")
                return CodeExecutionResult(False, '', error_msg, execution_time)
                
        except requests.exceptions.Timeout:
            error_msg = f'代码执行超时 ({timeout}秒)'
            logger.warning(error_msg)
            return CodeExecutionResult(False, '', error_msg, timeout)
            
        except requests.exceptions.ConnectionError:
            error_msg = '无法连接到SandboxFusion服务'
            logger.error(error_msg)
            return CodeExecutionResult(False, '', error_msg, time.time() - start_time)
            
        except Exception as e:
            error_msg = f'执行出错: {str(e)}'
            logger.error(error_msg)
            return CodeExecutionResult(False, '', error_msg, time.time() - start_time)

# 全局执行器实例
executor = SandboxFusionExecutor()

if __name__ == "__main__":
    pass

