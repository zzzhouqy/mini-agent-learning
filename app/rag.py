from pathlib import Path


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


def answer_with_context(question: str) -> dict:
    contexts = search_knowledge(question)

    return {
        "question": question,
        "contexts": contexts,
    }