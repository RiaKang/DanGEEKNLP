###
from konlpy.tag import Okt
from tqdm import tqdm
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from hyperparameters import *
from preprocessing import *
import numpy as np


### Word2Vec CBoW -> BoW형태에서 인근 "주위" 단어를 보고 "중심" 단어를 예측

# train_sents, train_labels, test_sents, test_labels = load_data()
# train_corpus, test_corpus = tokenize(train_sents, test_sents)
#
# CBoW_model = Word2Vec(sentences=train_corpus, vector_size=vector_size, window=window, min_count=min_count, epochs=num_epochs, workers=workers, sg=sg)

CBoW_model = Word2Vec.load("word2vec_cbow_nsmc.model")
def sentence_embedding_word2vec(model, sentence, dim=100):  # 문장에 존재하는 단어 embedding -> 벡터 처리
    sent_emb = np.zeros(dim)
    n = 0

    for word_i in sentence:
        try:  # Vocab 단어 사전에 존재하지 않는 단어는 예외처리
            word_emb = model.wv[word_i]
            sent_emb += word_emb
            n += 1
        except:
            pass

    # sentence embedding -> word embedding 평균치
    if n != 0:
        sent_emb = sent_emb / n

    return sent_emb


def x_cbow(train_corpus, test_corpus):
    train_x_cbow = list()
    test_x_cbow = list()

    # train과 test의 벡터 처리 된 단어들을 list 형태로 추가
    for train_sent in tqdm(train_corpus):
        train_x_cbow.append(sentence_embedding_word2vec(CBoW_model, train_sent))

    for test_sent in tqdm(test_corpus):
        test_x_cbow.append(sentence_embedding_word2vec(CBoW_model, test_sent))

    return train_x_cbow, test_x_cbow


# def train_x_cbow(train_corpus):
#     x_cbow = list()
#     # train의 벡터 처리 된 단어들을 list 형태로 추가
#     for train_sent in tqdm(train_corpus):
#         x_cbow.append(sentence_embedding_word2vec(CBoW_model, train_sent))
#
#     return x_cbow
#
# def test_x_cbow(test_corpus):
#     x_cbow = list()
#
#     for test_sent in tqdm(test_corpus):
#         x_cbow.append(sentence_embedding_word2vec(CBoW_model, test_sent))
#
#     return x_cbow

def real_x_cbow(real_corpus):
    x_cbow = list()

    print('Now on embedding')
    for real_sent in tqdm(real_corpus):
        x_cbow.append(sentence_embedding_word2vec(CBoW_model, real_sent))

    return x_cbow

