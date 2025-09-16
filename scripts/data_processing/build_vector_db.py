import os
from tqdm import tqdm
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_processor import process_pdf

def main():
    print("开始构建向量数据库...")
    
    # 处理PDF文档
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data/regulations/水利法律法规汇编（2023版）.pdf"
    print("处理PDF文档...")
    docs = process_pdf(str(pdf_path))
    print(f"生成 {len(docs)} 个文档块")
    
    # 创建向量存储
    print("创建向量存储...")
    # 初始化embedding模型
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 创建向量数据库
    Path('models').mkdir(exist_ok=True)
    db_path = project_root / "models/vector_db"
    print(f"正在向量化 {len(docs)} 个文档块...")
    # 分批处理文档
    batch_size = 10  # 每批处理10个文档
    vectorstore = None

    with tqdm(total=len(docs), desc="向量化进度", unit="块") as pbar:
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            if vectorstore is None:
                # 创建初始向量存储
                vectorstore = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=embeddings,
                    persist_directory=str(db_path),
                    collection_name="water_regulations"
                )
            else:
                # 添加文档到现有向量存储
                vectorstore.add_documents(batch_docs)
            
            # 更新进度条
            pbar.update(len(batch_docs))
    
    print("向量数据库创建完成")

if __name__ == "__main__":
    main()
