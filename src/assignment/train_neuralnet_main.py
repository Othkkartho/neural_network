from train_neuralnet_ass import two_train_start, three_train_start, four_train_start, five_train_start

if __name__ == "__main__":
    # 각 레이어 별로 hidden 값의 시작, 종료, 스킵할 값을 정해 넣으면 알아서 돌아감. 전부 돌려도 됨.
    two_train_start(15, 101, 5)
    three_train_start(20, 101, 10)
    four_train_start(20, 101, 20)
    five_train_start()
