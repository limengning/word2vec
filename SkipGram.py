import paddle
from paddle.fluid.layers.nn import embedding, pad
import paddle.nn
from paddle.nn import Embedding
import paddle.nn.functional as F
import paddle.optimizer
import numpy as np

import load


class SkipGram(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.5):
        super(SkipGram, self).__init__()
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


batch_size = 64
epoch_num = 1
embedding_size = 200
step = 0
learning_rate = 0.001


# paddle.set_device('gpu:0')

skip_gram_model = SkipGram(load.vocab_size, embedding_size)

adam = paddle.optimizer.Adam(
    learning_rate=learning_rate, parameters=skip_gram_model.parameters())

for center_words, target_words, label in load.build_batch(load.skipgram_dataset, batch_size, epoch_num):
    center_tensor = paddle.to_tensor(center_words)
    target_tensor = paddle.to_tensor(target_words)
    label_tensor = paddle.to_tensor(label)
    pre, loss = skip_gram_model(center_tensor, target_tensor, label_tensor)
    loss.backward()
    adam.step()
    adam.clear_grad()
    step += 1
    if step % 10000 == 0:
        load.get_similar_tokens('movie', 5, skip_gram_model.embedding.weight)
        load.get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
        load.get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)
