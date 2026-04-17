import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from logger import log_header, log_info, log_success

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / ".chroma_db"
print(DB_DIR)
print(Path(__file__).parent.parent / ".chroma_db")

load_dotenv(override=True)
# 配置 llm 相关内容
api_key = os.getenv("GLM_API_KEY")
base_url = os.getenv("GLM_BASE_URL")
model = os.getenv("GLM_MODEL")


embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),
)

log_success("嵌入模型已加载...")

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="langchain_docs",
    # 困扰我一天的问题找到了 向量数据库路径没有指定正确 streamlit 没有拿到metadata信息也是因为路径问题
    persist_directory=str(DB_DIR),
)

log_success("Chroma 已加载...")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))

# chat_model = ChatOpenAI(
#     model=model,
#     base_url=base_url,
#     api_key=api_key,
# )


chat_model = init_chat_model(
    model=os.getenv("OLLAMA_MODEL"), model_provider="ollama", temperature=0
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """这是一个检索文档的工具 用于回答用户关于Langchain相关的问题"""

    log_success("开始检索数据库~~~")
    retrieved_docs = vector_store.similarity_search(query, k=3)
    # vector_store.as_retriever()
    # retrieved_docs = vector_store.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        f'source: {doc.metadata.get("source","unknown")}\n\ncontent: {doc.page_content}'
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def llm_run(query: str) -> Dict[str, Any]:
    """
    通过rag检索 回答用户的问题

    Args:
        query: 用户向llm询问的问题
    Returns:
        字典内容:
            - answer: llm通过rag检索相关内容回答的内容
            - content: rag检索到的内容
    """

    sys_message = SystemMessage("""
        你是一个乐于助人而且非常有帮助的智能客服, 帮助解答用户关于 langchain 相关的问题
        你需要通过工具检索相关的知识
        你需要使用查阅到的资料进行回答
    """)

    """
        如果你不知道问题的答案你就说不知道
        如果你在检索方面遇到困难就说出来或者不知道怎么回答就说不知道 (模型问题本地模型还是不太行)
    """

    agent = create_agent(
        model=chat_model, tools=[retrieve_context], system_prompt=sys_message
    )

    # response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    response = agent.invoke({"messages": [ HumanMessage(query)]})

    answer = response["messages"][-1].content
    content = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                content.extend(message.artifact)

    return {
        "answer": answer,
        "content": content,
    }


if __name__ == "__main__":
    result = llm_run("什么是 deepagents")
    log_header("llm回答的内容")
    print(result.get('content'))
    log_success(result.get("answer"))
