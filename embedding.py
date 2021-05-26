import numpy as np
import paddle
from paddle.nn import Embedding

embedding = Embedding(100, 10, sparse=True)

corpus = paddle.to_tensor(np.array([1, 3, 5, 7]))
word_emb = embedding(corpus)

print(word_emb)
