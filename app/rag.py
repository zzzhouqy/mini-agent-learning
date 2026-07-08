from pathlib import Path
from app.agent import send_messages
from app.config import LLMConfig

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.md"


def load_knowledge() -> str:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{KNOWLEDGE_PATH}"
        )
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")

def split_paragraphs(text: str) -> list[str]:
    chunks = []
    current_chunk = []

    for paragraph in text.split("\n\n"):
        cleaned = paragraph.strip()
        if not cleaned:
            continue

        if cleaned.startswith("# ") and not cleaned.startswith("## "):
            continue

        if cleaned.startswith("## ") and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [cleaned]
        else:
            current_chunk.append(cleaned)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def search_knowledge(query: str, top_k: int = 3) -> list[str]:
    knowledge_text = load_knowledge()
    paragraphs = split_paragraphs(knowledge_text)
    keywords = query.lower().split()

    results = []
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in keywords):
            results.append(paragraph)

    return results[:top_k]


def format_contexts(contexts: list[str]) -> str:
    if not contexts:
        return "没有检索到相关资料。"

    formatted_contexts = []
    for index, context in enumerate(contexts, start=1):
        formatted_contexts.append(f"[资料 {index}]\n{context}")

    return "\n\n".join(formatted_contexts)


def build_rag_messages(question: str) -> list[dict]:
    contexts = search_knowledge(question)
    context_text = format_contexts(contexts)

    return [
        {
            "role": "system",
            "content": (
                "你是一个严谨的 RAG 问答助手。"
                "你只能基于用户提供的资料回答问题。"
                "如果资料不足，请回答：资料不足，无法基于本地知识库回答。"
                "不要编造资料中不存在的信息。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"可用资料：\n{context_text}"
            ),
        },
    ]

def answer_with_context(question: str) -> dict:
    contexts = search_knowledge(question)

    return {
        "question": question,
        "contexts": contexts,
    }


def rag_chat(question: str, config: LLMConfig) -> str:
    messages = build_rag_messages(question)
    message = send_messages(config, messages)

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型返回了空回答。")

    return content.strip()