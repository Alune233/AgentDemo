import re
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def clean_text(text: str) -> str:
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除页码等
    text = re.sub(r'第\s*\d+\s*页|页\s*共\s*\d+', '', text)
    return text.strip()

def process_pdf(pdf_path: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", "，", " "]
    )
    print(f"文档处理器初始化: chunk_size={1000}")
    print(f"处理PDF: {pdf_path}")
    
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")     
    # 加载PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"加载 {len(documents)} 页")
    # 清理文本
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    # 过滤空页面
    documents = [doc for doc in documents if doc.page_content.strip()]
    # 分块
    chunks = text_splitter.split_documents(documents)
    # 添加ID
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i:06d}"
    
    print(f"生成 {len(chunks)} 个文档块")
    return chunks

