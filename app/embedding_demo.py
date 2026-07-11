import math


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的维度必须一致。")
    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(vector_a, vector_b)
    )
    length_a = math.sqrt(sum(value**2 for value in vector_a))
    length_b = math.sqrt(sum(value**2 for value in vector_b))
    if length_a == 0 or length_b == 0:
        raise ValueError("向量长度不能为 0。")
    return dot_product / (length_a * length_b)


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    sentences = [
        "Pydantic 可以校验工具调用参数。",
        "模型参数格式错误时需要进行验证。",
        "今天天气很好。",
    ]
    embeddings = model.encode(sentences)

    print(embeddings.shape)
    print(embeddings[0][:5])
    related_score = cosine_similarity(
        embeddings[0].tolist(),
        embeddings[1].tolist(),
    )
    unrelated_score = cosine_similarity(
        embeddings[0].tolist(),
        embeddings[2].tolist(),
    )

    print(f"相关句子相似度：{related_score:.4f}")
    print(f"无关句子相似度：{unrelated_score:.4f}")