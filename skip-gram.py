import paddle
from paddle.fluid.layers.nn import pad
import paddle.nn
from paddle.nn import Embedding
import paddle.nn.functional as F


class SkipGram(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.5):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(num_embeddings=self.vocab_size,
                                   embedding_dim=self.embedding_size, weight_attr=paddle.nn.initializer.Uniform(
                                       low=-init_scale/embedding_size, high=init_scale*embedding_size
                                   ))
        self.embedding_out = Embedding(num_embeddings=self.vocab_size,
                                       embedding_dim=self.embedding_size, weight_attr=paddle.nn.initializer.Uniform(
                                           low=-init_scale/embedding_size, high=init_scale*embedding_size
                                       ))

    def forward(self, center_words, target_words, label):
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        word_sim = paddle.multiply(center_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)
        return pred, loss
