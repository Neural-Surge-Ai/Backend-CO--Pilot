import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ✅ Import your Pinecone retriever
from rag.retriever import retriever


# Load env (same path you're using)
load_dotenv(r"D:\Copilot\ai-copilot\.env")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

response_model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
grader_model = ChatOpenAI(model=MODEL_NAME, temperature=0)


# -----------------------------
# 1) Retriever Tool (NeuralSurge website)
# -----------------------------
def _format_docs(docs) -> str:
    """
    Make tool output informative + traceable using metadata (if exists in Pinecone).
    """
    blocks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("url") or ""
        title = d.metadata.get("title", "")
        section = d.metadata.get("section", "")

        meta = " | ".join([x for x in [src, title, section] if x]).strip()
        if meta:
            blocks.append(f"[{meta}]\n{d.page_content}")
        else:
            blocks.append(d.page_content)

    return "\n\n---\n\n".join(blocks)


@tool
def retrieve_neuralsurge_context(query: str) -> str:
    """
    Search and return information about NeuralSurge.ai from the Pinecone vector index.
    """
    docs = retriever.invoke(query)
    return _format_docs(docs)


retriever_tool = retrieve_neuralsurge_context


# -----------------------------
# 2) Node: generate_query_or_respond
# -----------------------------
def generate_query_or_respond(state: MessagesState):
    """
    LLM decides whether to call retriever tool or answer directly.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# -----------------------------
# 3) Node: grade_documents (conditional edge)
# -----------------------------
GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved context to a user question.\n\n"
    "Retrieved context:\n{context}\n\n"
    "User question:\n{question}\n\n"
    "If the context contains keyword(s) or semantic meaning that helps answer the question, "
    "grade it as relevant.\n"
    "Return only a binary score 'yes' or 'no'."
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, else 'no'")


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    score = (response.binary_score or "").strip().lower()
    return "generate_answer" if score == "yes" else "rewrite_question"


# -----------------------------
# 4) Node: rewrite_question
# -----------------------------
REWRITE_PROMPT = (
    "Rewrite the user question so it is more specific and easier to answer from a company website knowledge base.\n"
    "Preserve the user's intent.\n\n"
    "Original question:\n{question}\n\n"
    "Improved question:"
)


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=rewritten.content)]}


# -----------------------------
# 5) Node: generate_answer
# -----------------------------
GENERATE_PROMPT = (
    "You are an assistant answering questions about Neural Surge AI using ONLY the provided context.\n"
    "If the answer is not in the context, say: \"I don't know based on the website data.\" \n"
    "Keep the answer concise (max 5 sentences). If useful, use bullet points.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# -----------------------------
# 6) Assemble Graph (same as tutorial)
# -----------------------------
workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,  # if model answered directly
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


# -----------------------------
# 7) Run
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk (or type 'exit'): ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        for chunk in graph.stream({"messages": [{"role": "user", "content": q}]}):
            for node, update in chunk.items():
                print("\n=== Update from node:", node, "===\n")
                update["messages"][-1].pretty_print()
