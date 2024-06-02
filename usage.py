# from konlpy.tag import Okt
# from tqdm import tqdm
# from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, recall_score, f1_score
# from hyperparameters import *
from preprocessing import *
from embedding import *
import numpy as np
import joblib
import requests
import json


# Java 서버에서 Input 받기
# real_sents = ["안녕하세요 좋은 아침이에요", "좋은 하루 되세요", "아 진짜 너무 짜증나요", "적당히 하세요 씨발"]

# 받은 input 데이터로 AI 결과 출력
def test(model, real_sents):
    real_corpus = tokenize_real(real_sents)
    print(real_corpus)
    real_cbow = real_x_cbow(real_corpus)
    pred_real = model.predict(real_cbow)

    print(f'예측값 : {pred_real}')

    pred_real = int(pred_real[0])

    return pred_real

saved_model = joblib.load('./linear_cbow_model.pkl')

# bad_word = test(saved_model, real_sents)

from flask import Flask, request, render_template
import json
from werkzeug.utils import secure_filename
from socket import *

app = Flask(__name__)


# !! 주의 route 뒤에 경로는 위에 Spring에서 적은 경로와 같아야함 !!
@app.route('/receive_string', methods=['POST'])
def receive_string():
    # Spring으로부터 JSON 객체를 전달받음
    dto_json = request.get_json()

    dto_json['result'] = test(saved_model, dto_json['message'])

    # dto_json을 dumps 메서드를 사용하여 response에 저장
    response = json.dumps(dto_json, ensure_ascii=False)

    # Spring에서 받은 데이터를 출력해서 확인
    print(dto_json)

    # Spring으로 response 전달
    return response


# 0.0.0.0 으로 모든 IP에 대한 연결을 허용해놓고 포트는 8080로 설정
if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)
