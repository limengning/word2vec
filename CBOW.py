import paddle
from paddle.fluid.layers.nn import embedding, pad
import paddle.nn
from paddle.nn import Embedding
import paddle.nn.functional as F
import paddle.optimizer
import numpy as np

import load


class CBOW(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.5):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(num_embeddings=self.vocab_size,
                                   embedding_dim=self.embedding_size, weight_attr=paddle.nn.initializer.Uniform(
                                       low=-init_scale, high=init_scale
                                   ))
        self.embedding_out = Embedding(num_embeddings=self.vocab_size,
                                       embedding_dim=self.embedding_size, weight_attr=paddle.nn.initializer.Uniform(
                                           low=-init_scale, high=init_scale
                                       ))

    def forward(self, context_words, center_words, label):
        context_words_emb = self.embedding(context_words)
        center_words_emb = self.embedding_out(center_words)

        word_sim = paddle.multiply(context_words_emb, center_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)
        return pred, loss


batch_size = 64
epoch_num = 1
embedding_size = 200
step = 0
learning_rate = 0.001


# paddle.set_device('gpu:0')

cbow_model = CBOW(load.vocab_size, embedding_size)

adam = paddle.optimizer.Adam(
    learning_rate=learning_rate, parameters=cbow_model.parameters())

for context_words, center_words, label in load.build_batch(load.cbow_dataset, batch_size, epoch_num):
    context_tensor = paddle.to_tensor(context_words)
    center_tensor = paddle.to_tensor(center_words)
    label_tensor = paddle.to_tensor(label)
    pre, loss = cbow_model(context_tensor, center_tensor, label_tensor)
    loss.backward()
    adam.step()
    adam.clear_grad()
    step += 1
    if step % 10000 == 0:
        load.get_similar_tokens('movie', 5, cbow_model.embedding.weight)
        load.get_similar_tokens('one', 5, cbow_model.embedding.weight)
        load.get_similar_tokens('chip', 5, cbow_model.embedding.weight)
