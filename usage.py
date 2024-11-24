# from konlpy.tag import Okt
# from tqdm import tqdm
# from datasets import load_dataset
# from sklearn.metrics import accuracy_score, recall_score, f1_score
# from hyperparameters import *
# from preprocessing import *
# from embedding import *
import numpy as np
# import joblib
# import requests
# import json
# from math import *

# Java 서버에서 Input 받기

# 받은 input 데이터로 AI 결과 출력
# def test(model, real_sents):
#     real_corpus = tokenize_real(real_sents)
#     print(real_corpus)
#     real_cbow = real_x_cbow(real_corpus)
#     pred_real = model.predict(real_cbow)
#
#     print(f'예측값 : {pred_real}')
#
#     pred_real = int(pred_real[0])
#
#     return pred_real
#
# saved_model = joblib.load('./linear_cbow_model.pkl')

from flask import Flask, request, render_template
import json

app = Flask(__name__)

#
from openai import OpenAI
client = OpenAI()
# !! 주의 route 뒤에 경로는 위에 Spring에서 적은 경로와 같아야함 !!
@app.route('/receive_string', methods=['POST'])
def receive_string():
    # GPT openAI 응답값으로 응답하기
    dto_json = request.get_json()

    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::AUaBNjLL",
        messages=[
            {"role": "system", "content": "The model determines whether a sentence is immoral."},
            {"role": "user", "content": dto_json['message']}
        ]
    )

    ai_result = completion.choices[0].message.content
    if (ai_result == 'true'):
        dto_json['result'] = 0
    elif(ai_result == 'false'):
        dto_json['result'] = 1
    else:
        dto_json['result'] = 0
        print('Cant determine True or False')

    response = json.dumps(dto_json, ensure_ascii=False)
    return response

    ################################# 원래 학습 return 방식
    # "message", "result" 형식
    # Spring으로부터 메세지 JSON 객체를 전달받음
    #dto_json = request.get_json()

    #dto_json['result'] = test(saved_model, dto_json['message'])

    # dto_json을 dumps 메서드를 사용하여 response에 저장
    #response = json.dumps(dto_json, ensure_ascii=False)

    # Spring에서 받은 데이터를 출력해서 확인
    #print(dto_json)

    # Spring으로 response 전달
    #return response

# @app.route('/receive_string2', methods=['POST'])
# def receive_string2():
#     dto_json = request.get_json()
#
#     #"message", "result" 형식
#     #Spring으로부터 메세지 JSON 객체를 전달받음
#     dto_json = request.get_json()
#
#     dto_json['result'] = test(saved_model, dto_json['message'])
#
#     #dto_json을 dumps 메서드를 사용하여 response에 저장
#     response = json.dumps(dto_json, ensure_ascii=False)
#
#     #Spring에서 받은 데이터를 출력해서 확인
#     print(dto_json)
#
#     #Spring으로 response 전달
#     return response

class Similarity:
    memberId=''
    similarity = 0

    def __init__(self, id, sim):
        self.memberId = id
        self.similarity = sim

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality / float(union_cardinality)



@app.route('/receive_similarity', methods=['POST'])
def receive_similiarity():
    # Spring으로부터 유사도 JSON 객체를 전달받음
    # "my_vector", "dict_list_vector", "clean_vector" 벡터 내부에는 기상 시간, 취침 시간, 취미 벡터 12개, 청결도 벡터 10개
    # 기상시간 : 5~15, 취침시간 : 19~29, 취미 벡터, 청경도 벡터는 0,1로 구분
    dto_json = request.get_json()
    my_vector = dto_json["myVector"]
    dict_list_vector = dto_json.get("memberVectors")
    print(dict_list_vector)

    ret_vec = dict({"memberId": [], "similarity": []})
    ret_vec_add = dict({"memberId": [], "similarity": []})
    ret_list = []
    for list in dict_list_vector:
        vec = list.get("vector")
        awake_hour_diff = (10 - abs(my_vector[0] - vec[0]))*0.05
        sleep_hour_diff = (10 - abs(my_vector[1] - vec[1]))*0.05
        hobby_similarity = cosine_similarity(my_vector[2:14], vec[2:14])
        similarity = (hobby_similarity + awake_hour_diff + sleep_hour_diff + cosine_similarity(my_vector[14:], vec[14:]))
        memId = list.get("memberId")

        ret_list.append(Similarity(memId, similarity))

    ret_dict = dict()
    ret_dict['result'] = [ob.__dict__ for ob in ret_list]
    sorted_dict_json = json.dumps(ret_dict, ensure_ascii=False)
    print(sorted_dict_json)

    return sorted_dict_json

# 0.0.0.0 으로 모든 IP에 대한 연결을 허용해놓고 포트는 8080로 설정
if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)
