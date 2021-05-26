
import requests


def download():
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    req = requests.get(corpus_url)
    with open("./text8.txt", "wb") as f:
        for chunk in req.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    f.close


download()
