import pytest

from app.embedding_demo import cosine_similarity


def test_cosine_similarity():
    result = cosine_similarity([1.0, 0.0], [0.8, 0.2])

    assert result == pytest.approx(0.9701425001453318)


def test_rejects_different_dimensions():
    with pytest.raises(ValueError, match="两个向量的维度必须一致"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_rejects_zero_vector():
    with pytest.raises(ValueError, match="向量长度不能为 0"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])