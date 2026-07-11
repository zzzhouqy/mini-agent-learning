from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def encode_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts)

    return embeddings.tolist()
def semantic_scores(query: str, texts: list[str]) -> list[float]:
    model = get_embedding_model()
    embeddings = model.encode([query, *texts])
    similarities = model.similarity(
        embeddings[:1],
        embeddings[1:],
    )

    return similarities[0].tolist()