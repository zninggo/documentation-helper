import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv(override=True)

# 配置 llm 相关内容
api_key = os.getenv("GLM_API_KEY")
base_url = os.getenv("GLM_BASE_URL")
model = os.getenv("GLM_MODEL")

# Configure SSL context to use certifi certificates 配置 SSL 上下文以使用 certifi 证书
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),
)

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="langchain_docs",
    persist_directory="./.chroma_db",
)

log_success("向量数据库创建/加载完成...")

tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False,
)

tavily_crawl_tool = TavilyCrawl()

tavily_map = TavilyMap()

log_success("依赖加载完成...")


# python_langchain_entry = "https://opencode.ai/docs/zh-cn"
# python_langchain_entry = "https://docs.python.org/zh-cn/3.11/library/"
# python_langchain_entry = "https://nodejs.org/docs/latest/api/"
python_langchain_entry = "https://docs.langchain.com/oss/python/langchain/overview"


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """异步批量处理文档."""
    log_header("向量存储")
    log_info(
        f"📚 VectorStore 索引：准备将 {len(documents)} 文档添加到向量存储",
        Colors.DARKCYAN,
    )

    #     分批次
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(f"📦 VectorStore 索引：分成 {len(batches)} 批，每批 {batch_size} 个文档")

    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        """批次处理添加到向量数据库"""
        try:

            await vector_store.aadd_documents(batch)
            log_success(
                f"VectorStore 索引：已成功添加批次 {batch_num}/{len(batches)}（{len(batch)} 文档）"
            )

        except Exception as e:
            log_error(f"VectorStore 索引：无法添加批次 {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore 索引：所有批次均已成功处理! ({successful}/{len(batches)})"
        )
    else:
        log_warning(f"矢量存储索引：已成功处理 {successful}/{len(batches)} 个批次")


async def main():
    """主要的异步函数来协调整个过程。"""
    log_header("文档爬取")

    log_info("🗺️ TavilyCrawl：开始抓取文档站点", color=Colors.PURPLE)

    response = None
    try:
        response = tavily_crawl_tool.invoke(
            {
                "url": python_langchain_entry,
                "max_depth": 5,
                "max_breadth": 100,
                "limit": 500,
            }
        )

    except Exception as e:

        print(e)

    log_success(
        f"Tavily Crawl: 从 {python_langchain_entry} 入口处 爬取到 {len(response['results'])} 个链接入口 "
    )

    all_docs = [
        Document(
            page_content=result.get("raw_content", ""),
            metadata={"source": result.get("url")},
        )
        for result in response.get("results")
        if result.get("raw_content") is not None
    ]

    # print(all_docs)
    # Split documents into chunks
    log_header("文档分块")

    chunk_overlap = 400
    chunk_size = 2000
    log_info(
        f"✂️  文本分块: 处理具有 {chunk_size} 块大小和 {chunk_overlap} 重叠的 {len(all_docs)} 文档",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(all_docs)
    log_success(
        f"文本分块: Created {len(chunks)} chunks from {len(all_docs)} documents"
    )

    # 异步处理文档 存储向量数据库
    await index_documents_async(chunks, 500)

    log_header("任务完成")
    log_success("🎉 文档存储向量数据库完成!")
    log_info("📊 总结:", Colors.BOLD)
    log_info(f"   • 提取的文件: {len(all_docs)}")
    log_info(f"   • 创建块: {len(chunks)}")


if __name__ == "__main__":
    asyncio.run(main())
