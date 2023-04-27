from train_neuralnet_ass import two_train_start, three_train_start, four_train_start, five_train_start

# 입력값 바꿀려면 이거랑, dataset/mnist_ass.py의 img_size 랑 바꾸면 됨. (변경 추천 수치, 392, 196)
# input_size 바꾸고, mnist_ass.pkl이 만들어져 있다면 그거 없애고, 다시 돌릴것
INPUT_SIZE = 196

if __name__ == "__main__":
    # 각 레이어 별로 hidden 값의 시작, 종료, 스킵할 값을 정해 넣으면 알아서 돌아감. 전부 돌려도 됨.
    two_train_start(INPUT_SIZE, 15, 101, 5)
    three_train_start(INPUT_SIZE, 20, 101, 20)
    four_train_start(INPUT_SIZE, 25, 101, 25)
    five_train_start(INPUT_SIZE)
