import os
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("VectorSearch")

class VectorSearchTool:
    def __init__(self):
        self.vectorstore = None
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # 从环境变量读取配置
            db_path = str(Path(__file__).parent.parent.parent/"models/vector_db")
            collection_name = "water_regulations"
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
            
            # 初始化embedding
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 加载向量存储
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # 检查数据
            count = self.vectorstore._collection.count()
            if count > 0:
                logger.info(f"向量搜索工具初始化成功，文档数: {count}")
            else:
                logger.warning("向量数据库为空")
                
        except Exception as e:
            logger.error(f"向量搜索工具初始化失败: {e}")
            self.vectorstore = None
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise ValueError("向量搜索工具未初始化")
        
        logger.debug(f"搜索查询: {query}")
        
        # 相似度搜索
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        logger.info(f"搜索完成，返回 {len(formatted_results)} 个结果")
        return formatted_results
    
    def is_available(self) -> bool:
        return self.vectorstore is not None

# 全局搜索工具实例
search_tool = VectorSearchTool()

if __name__ == "__main__":
    pass
