# https://github.com/zninggo/documentation-helper/blob/main/main.py
from typing import Any, List

import streamlit as st

from backend.core import llm_run
from logger import log_info


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain 文档助手", layout="centered")
st.title("LangChain 文档助手")


with st.sidebar:
    st.subheader("Session")
    if st.button("清除聊天记录", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "关于LangChain文档的任何问题都可以问我。我会检索相关背景并引用资料。",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("提问关于LangChain的问题…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("检索文档并生成答案…"):
                result = llm_run(prompt)
                answer = result.get("answer", "").strip() or "没有回复~"
                sources = _format_sources(result.get("content", []))
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
