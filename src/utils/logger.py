import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, level: str = "INFO", use_timestamp: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if use_timestamp:
        # 基于项目根目录和时间戳创建日志文件
        project_root = Path(__file__).parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"agent_{timestamp}.log"
        log_path = project_root / "logs" / log_filename
        
        # 确保日志目录存在
        log_path.parent.mkdir(parents=True, exist_ok=True)    
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 记录日志文件位置
        logger.info(f"日志文件: {log_path}")
    
    return logger

# 项目默认日志器
project_logger = setup_logger(name="AgentDemo", level="INFO")