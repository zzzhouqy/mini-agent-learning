from pathlib import Path
from app.agent import send_messages
from app.config import LLMConfig
from app.embeddings import semantic_scores

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.md"
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"

def load_knowledge() -> str:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{KNOWLEDGE_PATH}"
        )
    return KNOWLEDGE_PATH.read_text(encoding="utf-8")



def load_documents() -> list[dict[str, str]]:
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    documents = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        documents.append(
            {
                "source": str(path.relative_to(KNOWLEDGE_DIR.parent.parent)),
                "text": path.read_text(encoding="utf-8"),
            }
        )

    return documents
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


def build_context(chunk: str, score: int = 0) -> dict[str, str]:
    lines = chunk.splitlines()
    title = lines[0].lstrip("# ").strip()
    content = "\n".join(lines[1:]).strip()

    return {
        "source": "data/knowledge.md",
        "title": title,
        "content": content,
        "score": str(score),
    }

def build_context_from_document(
    document: dict[str, str],
    chunk: str,
    score: int = 0,
) -> dict[str, str]:
    document_lines = document["text"].splitlines()
    title = document_lines[0].lstrip("# ").strip()
    content = chunk.strip()

    return {
        "source": document["source"],
        "title": title,
        "content": content,
        "score": str(score),
    }
def score_chunk(chunk: str, keywords: list[str]) -> int:
    chunk_lower = chunk.lower()
    score = 0

    for keyword in keywords:
        if keyword in chunk_lower:
            score += 1

    return score
def search_knowledge(query: str, top_k: int = 3) -> list[dict[str, str]]:
    documents = load_documents()
    keywords = query.lower().split()

    scored_results = []
    for document in documents:
        chunks = split_paragraphs(document["text"])
        for chunk in chunks:
            score = score_chunk(chunk, keywords)
            if score > 0:
                scored_results.append(
                    (
                        score,
                        build_context_from_document(
                            document,
                            chunk,
                            score,
                        ),
                    )
                )

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = []
    for _, context in scored_results[:top_k]:
        results.append(context)

    return results

def semantic_search_knowledge(
    query: str,
    top_k: int = 3,
    min_score: float = 0.2,
) -> list[dict[str, str]]:
    candidates = []

    for document in load_documents():
        for chunk in split_paragraphs(document["text"]):
            candidates.append(
                build_context_from_document(document, chunk)
            )

    if not candidates:
        return []

    texts = [
        f"{context['title']}\n{context['content']}"
        for context in candidates
    ]
    scores = semantic_scores(query, texts)

    scored_results = []
    for score, context in zip(scores, candidates):
        if score >= min_score:
            context["score"] = str(score)
            scored_results.append((score, context))

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        context
        for _, context in scored_results[:top_k]
    ]
def format_contexts(contexts: list[dict[str, str]]) -> str:
    if not contexts:
        return "没有检索到相关资料。"

    formatted_contexts = []
    for index, context in enumerate(contexts, start=1):
        formatted_contexts.append(
            (
                f"[资料 {index}]\n"
                f"来源：{context['source']}\n"
                f"标题：{context['title']}\n"
                f"内容：\n{context['content']}"
            )
        )

    return "\n\n".join(formatted_contexts)


def build_rag_messages(question: str) -> list[dict]:
    contexts = semantic_search_knowledge(question)
    context_text = format_contexts(contexts)

    return [
        {
            "role": "system",
            "content": (
                "你是一个严谨的 RAG 问答助手。"
                "你只能基于用户提供的资料回答问题。"
                "如果资料不足，请回答：资料不足，无法基于本地知识库回答。"
                "不要编造资料中不存在的信息。"
                "如果资料中包含来源和标题，请在回答末尾用“依据：”列出来源和标题。"
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