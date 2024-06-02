seed = 42
num_epochs = 50  # 반복할 epoch 수
vector_size = 100  # 단어 vector 크기
window = 5  # 중심 단어를 기준으로 양 옆 몇 번째 단어까지 볼 것인가
min_count = 3  # 최소 n번 이상 등장한 단어에 대해서 적용
workers = 4
sg = 0  # 0 = CBoW / 1 = Skip-Gram