import random
import math
import numpy as np


def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()
    return corpus


corpus = load_text8()
print(corpus[:500])


def data_preprocess(corpus):
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus


corpus = data_preprocess(corpus)
print(corpus[:50])


def build_dict(corpus):
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(),
                            key=lambda x: x[1], reverse=True)
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    for word, freq in word_freq_dict:
        word_id = len(word2id_dict)
        word2id_dict[word] = word_id
        word2id_freq[word_id] = freq
        id2word_dict[word_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)

vocab_size = len(word2id_freq)


def convert_corpus_to_id(corpus, word2id_dict):
    corpus = [word2id_dict[w] for w in corpus]
    return corpus


corpus = convert_corpus_to_id(corpus, word2id_dict)


def subsampling(corpus, word2id_freq):
    def discard(word_id):
        return random.uniform(
            0, 1) < (1 - math.sqrt(1e-4 / word2id_freq[word_id] * len(corpus)))
    corpus = [c for c in corpus if not discard(c)]
    return corpus


corpus = subsampling(corpus, word2id_freq)


def build_data(corpus, max_window_size=3, negative_sample_num=4):
    dataset = []

    for center_word_idx in range(len(corpus)):
        window_size = random.randint(1, max_window_size)
        start_idx = max(0, center_word_idx - window_size)
        end_idx = min(len(corpus), center_word_idx + window_size)
        center_word = corpus[center_word_idx]
        positive_word_candidates = [corpus[idx] for idx in range(
            start_idx, end_idx) if not idx == center_word_idx]

        for positive_word in positive_word_candidates:
            dataset.append((center_word, positive_word, 1))
            i = 0
            while i < negative_sample_num:
                negative_word = random.randint(1, vocab_size - 1)
                if negative_word not in positive_word_candidates:
                    dataset.append((center_word, negative_word, 0))
                    i += 1
    return dataset


corpus_light = corpus[:int(len(corpus) * 0.2)]
dataset = build_data(corpus_light)


def build_batch(dataset, batch_size, epoch_num):
    center_word_batch = []
    target_word_batch = []
    label_batch = []

    for epoch in range(epoch_num):
        random.shuffle(dataset)
        for center_word, target_word, label in dataset:
            center_word_batch.append(center_word)
            target_word_batch.append(target_word)
            label_batch.append(label)
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch), np.array(target_word_batch), np.array(label_batch)
                center_word_batch = []
                target_word_batch = []
                label_batch = []
    if len(center_word_batch) > 0:
        yield np.array(center_word_batch), np.array(target_word_batch), np.array(label_batch)
