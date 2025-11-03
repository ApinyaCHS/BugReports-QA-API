
"""
FastAPI wrapper for your LangGraph + FAISS + Ollama pipeline.

Quick start:
  1) Make sure Ollama is running and the specified models are pulled:
       ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M
       ollama pull alibayram/Qwen3-30B-A3B-Instruct-2507:latest
  2) Ensure your FAISS index exists at /workspace/internal/vectorstores/bug_reports_index
  3) Install deps and run the API:
       pip install -r requirements.txt
       uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  4) Test:
       curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What are the issues reported on Pagination?"}'
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain / LangGraph imports
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- Configuration via env (with sensible defaults to match your notebook) ---
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "dengcao/Qwen3-Embedding-8B:Q5_K_M")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL", "alibayram/Qwen3-30B-A3B-Instruct-2507:latest")
VECTORSTORE_PATH   = os.getenv("VECTORSTORE_PATH", "/workspace/internal/vectorstores/bug_reports_index")

# --- Initialize embeddings, vector store, and LLM ---
try:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to init OllamaEmbeddings [{OLLAMA_EMBED_MODEL}]: {e}")

try:
    vector_store = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load FAISS index from '{VECTORSTORE_PATH}'. "
        f"Make sure the index exists and is built with the same embedding model. Error: {e}"
    )

try:
    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        temperature=0.0,
        num_predict=2048,
    )
except Exception as e:
    raise RuntimeError(f"Failed to init ChatOllama [{OLLAMA_CHAT_MODEL}]: {e}")

# --- Define retrieve tool ---
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve information related to a query from the FAISS vector database.
    Returns a serialized string (for the LLM) + the raw docs (artifact).
    """
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# --- Routing LLM step: decide to call retrieve() or not ---
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
SYSTEM_ROUTER_PROMPT = """
You are a reasoning router that decides whether to:
(1) call the `retrieve` tool to search for relevant information, or
(2) directly answer the user based on your own knowledge and context.

Guidelines:
- If the user’s question clearly requires external or document-based information (e.g., “Find”, “Search”, “Summarize reports”, “Retrieve”, “Show me”, “List recent”, “What are the latest bugs”), then CALL the `retrieve` tool.
- If the user asks general knowledge questions unrelated to IT or bug reports, then respond directly.
- If the query mixes both (e.g., “Explain based on the report” or “Compare last week’s and this week’s summaries”), CALL the `retrieve` tool first.


Return your reasoning implicitly through your choice — either call `retrieve()` or respond directly.
"""

def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    messages = [{"role": "system", "content": SYSTEM_ROUTER_PROMPT}] + state["messages"]
    response = llm_with_tools.invoke(messages)

    if getattr(response, "tool_calls", None):
        print("→ Router decided to QUERY via retrieve()")
    else:
        print("→ Router decided to RESPOND directly")

    return {"messages": [response]}

# --- Generation step: produce final JSON answer using any retrieved context ---


def generate(state: MessagesState):
    """Generate final answer using retrieved documents and routing context."""
    # === Step 1: Collect recent ToolMessages (retrieved docs) ===
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = list(reversed(recent_tool_messages))

    # === Step 2: Format retrieved content ===
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    # === Step 3: Compose system prompt ===
    system_message_content = (
    "You are an intelligent assistant specializing in IT systems, product feedback, "
    "and bug triage.\n"
    "Use the retrieved context below to answer the user's question.\n\n"
    f"=== Retrieved Context Start ===\n{docs_content}\n=== Retrieved Context End ===\n\n"
    "for example of retrieved context could be:\n"
    "Incorrect Search Results for Acronyms. Searching for acronyms (e.g., 'AI') returns irrelevant documents that contain the individual letters but not the acronym itself."
    "first is the title, second is the description.\n\n"
    "Answering Guidelines:\n"
    "If the user's question is about IT systems, bug reports, internal feedback, "
    "or technical documentation, respond strictly in JSON with the following schema:\n"
    "{'Reported issues': 'Title of the issue', 'Description': 'Detailed description of the issue'}\n" 
    "for example answer could be:\n"
    "{'Reported issues': 'Incorrect Search Results for Acronyms', 'Description': 'Searching for acronyms (e.g., AI) returns irrelevant documents that contain the individual letters but not the acronym itself.'}"

    "If you cannot find relevant information in the context, respond with: {'Reported issues': '', 'Description': 'No relevant information found.'}"
)

    # === Step 4: Collect conversation messages ===
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # === Step 5: Build final prompt and generate ===
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)

    return {"messages": [response]}

# --- Build graph ---
graph_builder = StateGraph(MessagesState)
tools = ToolNode([retrieve])
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# --- FastAPI app ---
app = FastAPI(title="Bug Reports QA API", version="1.0.0")

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": OLLAMA_EMBED_MODEL, "chat_model": OLLAMA_CHAT_MODEL}

@app.post("/ask")
def ask(req: AskRequest):
    try:
        # Run the LangGraph program
        result = graph.invoke({"messages": [{"role": "user", "content": req.question}]})
        messages = result.get("messages", [])
        final_msg = messages[-1] if messages else None
        content = getattr(final_msg, "content", "") if final_msg else ""

        # Also include naive top-5 sources (directly from vector store) for transparency
        top_docs = vector_store.similarity_search(req.question, k=5)
        sources = [
            {
                "metadata": d.metadata,
                "preview": d.page_content[:300]
            } for d in top_docs
        ]

        return {"answer": content, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

# Optional root
@app.get("/")
def root():
    return {
        "message": "Welcome to the Bug Reports QA API. POST to /ask with {'question': '...'}",
        "models": {"embedding": OLLAMA_EMBED_MODEL, "model llm": OLLAMA_CHAT_MODEL},
        "vectorstore_path": VECTORSTORE_PATH,
    }
