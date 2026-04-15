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

from logger import Colors, log_header, log_info, log_success

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

tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False,
)

tavily_crawl_tool = TavilyCrawl()

tavily_map = TavilyMap()

print("依赖加载完成...")


python_langchain_entry = "https://docs.langchain.com/oss/python/langchain/overview"


async def main():
    """主要的异步函数来协调整个过程。"""
    log_header("文档摄取管道")

    log_info("🗺️ TavilyCrawl：开始抓取文档站点", color=Colors.PURPLE)

    response = tavily_crawl_tool.invoke(
        {
            "url": python_langchain_entry,
            "depth": "advanced",
            "max_depth": 1,
        }
    )

    log_success(
        f"Tavily Crawl: 从 {python_langchain_entry} 入口处 爬取到 {len(response['results'])} 个链接入口 "
    )

    all_docs = [
        Document(
            page_content=result.get("raw_content"),
            metadata={"source": result.get("url")},
        )
        for result in response.get("results")
    ]

    print(all_docs)


if __name__ == "__main__":
    asyncio.run(main())
