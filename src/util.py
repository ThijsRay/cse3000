from numpy import ndarray, float32, dot
from numpy.linalg import norm


def cosine_similarity(a: ndarray, b: ndarray) -> float32:
    """Calculate the cosine similarity between two vectors. The definition is based on
     https://en.wikipedia.org/wiki/Cosine_similarity and the snippet of code is based on
     https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists/43043160#43043160"""
    return dot(a, b) / (norm(a) * norm(b))

