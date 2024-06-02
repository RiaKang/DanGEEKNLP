###
from konlpy.tag import Okt
from tqdm import tqdm
from datasets import load_dataset, Dataset
from hanspell import spell_checker
from pykospacing import Spacing

# dataset 불러오기
def load_data():

    # train_dataset = Dataset.from_file("traindata-00000-of-00001.arrow")
    # test_dataset = Dataset.from_file("testdata-00000-of-00001.arrow")
    #
    # train_sents = [train_dataset[idx]["comments"] for idx in range(len(train_dataset))]
    # train_labels = [train_dataset[idx]["hate"] for idx in range(len(train_dataset))]
    # test_sents = [test_dataset[idx]["comments"] for idx in range(len(test_dataset))]
    # test_labels = [test_dataset[idx]["hate"] for idx in range(len(test_dataset))]

    # dataset = load_dataset("kor_hate")
    #
    # train_sents = [dataset["train"][idx]["comments"] for idx in range(len(dataset["train"]))]
    # train_labels = [dataset["train"][idx]["hate"] for idx in range(len(dataset["train"]))]
    # test_sents = [dataset["test"][idx]["comments"] for idx in range(len(dataset["test"]))]
    # test_labels = [dataset["test"][idx]["hate"] for idx in range(len(dataset["test"]))]

    dataset = load_dataset("nsmc")

    train_sents = [dataset["train"][idx]["document"] for idx in range(len(dataset["train"]))]
    train_labels = [dataset["train"][idx]["label"] for idx in range(len(dataset["train"]))]
    test_sents = [dataset["test"][idx]["document"] for idx in range(len(dataset["test"]))]
    test_labels = [dataset["test"][idx]["label"] for idx in range(len(dataset["test"]))]

    return train_sents, train_labels, test_sents, test_labels

def spell_spacing(origin_sent):
    spacing = Spacing()
    # okt = Okt()
    for sent_i in origin_sent:
        # new_sent_i = spell_checker.check(sent_i).checked
        new_sent_i = spacing(sent_i)
        # words_i = [w for w, p in okt.pos(new_sent_i)]

    return new_sent_i

# # train과 test dataset 의 문장들을 tokenizing
def tokenize(train_sents, test_sents):
    # stopwords (불용어) 정의
    stopwords = ["하다", "한", "에", "와", "자", "과", "걍", "잘", "좀", "는", "의", "가", "이", "은", "들"]

    okt = Okt()

    train_corpus = []
    for sent_i in tqdm(train_sents):
        # 한글이 아닌 글자 제거
        new_sent_i = sent_i.replace("[^가-힣ㄱ-하-ㅣ]", "")  # 한글이 아닌 글자에 대해 ""로 없애줌
        # spell_spacing(new_sent_i)
        tokens = okt.morphs(new_sent_i, stem=True)  # 어간 추출
        tokens = [word_i for word_i in tokens if word_i not in stopwords]  # 불용어가 아닌 글자는 token에 추가

        train_corpus.append(tokens)

    test_corpus = []
    for sent_i in tqdm(test_sents):
        # 한글이 아닌 글자 제거
        new_sent_i = sent_i.replace("[^가-힣ㄱ-하-ㅣ]", "")  # 한글이 아닌 글자에 대해 ""로 없애줌
        # spell_spacing(new_sent_i)
        tokens = okt.morphs(new_sent_i, stem=True)  # 어간 추출
        tokens = [word_i for word_i in tokens if word_i not in stopwords]  # 불용어가 아닌 글자는 token에 추가

        test_corpus.append(tokens)

    return train_corpus, test_corpus

def tokenize_real(real_sents):
    # stopwords (불용어) 정의
    stopwords = ["하다", "한", "에", "와", "자", "과", "걍", "잘", "좀", "는", "의", "가", "이", "은", "들"]

    okt = Okt()

    print('Now on tokenizing')
    real_corpus = []
    for sent_i in tqdm(real_sents):
        # 한글이 아닌 글자 제거
        new_sent_i = sent_i.replace("[^가-힣ㄱ-하-ㅣ]", "")  # 한글이 아닌 글자에 대해 ""로 없애줌
        # spell_spacing(new_sent_i)
        tokens = okt.morphs(new_sent_i, stem=True)  # 어간 추출
        tokens = [word_i for word_i in tokens if word_i not in stopwords]  # 불용어가 아닌 글자는 token에 추가

        real_corpus.append(tokens)

    return real_corpus

# def tokenize_train(train_sents):
#     # stopwords (불용어) 정의
#     stopwords = ["하다", "한", "에", "와", "자", "과", "걍", "잘", "좀", "는", "의", "가", "이", "은", "들"]
#
#     okt = Okt()
#
#     train_corpus = []
#     for sent_i in tqdm(train_sents):
#         # 한글이 아닌 글자 제거
#         new_sent_i = sent_i.replace("[^가-힣ㄱ-하-ㅣ]", "")  # 한글이 아닌 글자에 대해 ""로 없애줌
#         # spell_spacing(new_sent_i)
#         tokens = okt.morphs(new_sent_i, stem=True)  # 어간 추출
#         tokens = [word_i for word_i in tokens if word_i not in stopwords]  # 불용어가 아닌 글자는 token에 추가
#
#         train_corpus.append(tokens)
#
#     return train_corpus
#
# def tokenize_test(test_sents):
#     # stopwords (불용어) 정의
#     stopwords = ["하다", "한", "에", "와", "자", "과", "걍", "잘", "좀", "는", "의", "가", "이", "은", "들"]
#
#     okt = Okt()
#
#     test_corpus = []
#     for sent_i in tqdm(test_sents):
#         # 한글이 아닌 글자 제거
#         new_sent_i = sent_i.replace("[^가-힣ㄱ-하-ㅣ]", "")  # 한글이 아닌 글자에 대해 ""로 없애줌
#         # spell_spacing(new_sent_i)
#         tokens = okt.morphs(new_sent_i, stem=True)  # 어간 추출
#         tokens = [word_i for word_i in tokens if word_i not in stopwords]  # 불용어가 아닌 글자는 token에 추가
#
#         test_corpus.append(tokens)
#
#     return test_corpus

