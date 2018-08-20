# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec

#訓練word2vec
def main():
    print("good")
    sentences = word2vec.Text8Corpus("D:\\NTUST\\人工智慧\\final\\csv\\sentence_retmoveAlphanum_seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)

    print("finish")
    # Save our model.
    model.save("D:\\NTUST\\人工智慧\\final\\csv\\code250.model.bin")

    print("saved")

if __name__ == "__main__":
    main()
