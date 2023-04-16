# coding: utf-8
import os
import sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist_ass import load_mnist
from two_layer_net import TwoLayerNet
from three_layer_net import ThreeLayerNet
from four_layer_net import FourLayerNet
from five_layer_net import FiveLayerNet


def two_layer(inputSize=784, hidden1=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 2~4 계층까지 데이터 생성해 저장. 4 계층 이후는 정확도 올라가는 속도가 너무 늦거나, 안올라 가기 때문에 이후는 안만듬. 2~4 까지만 하면 될 듯, 레이어 노드 수를 변경하려면
    network = TwoLayerNet(input_size=inputSize, hidden_size=hidden1, output_size=10)

    # 하이퍼파라미터
    iters_num = 10000  # 반복 횟수 설정
    train_size = x_train.shape[0]
    batch_size = 100  # 미니배치 크기
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1 에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        # grad = network.numerical_gradient(x_batch, t_batch)   # 수치 미분 방식 <- 원래는 이거 써야함. 그런데 속도가 상당히 느림. 만약 하다가 열받으면 그냥 아래꺼 쓸것
        grad = network.gradient(x_batch, t_batch)  # 오차역전파 방식(속도 빠름)

        # 계층 별로 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    hidden_list = [hidden1]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    # 그래프 그리기
    grape1(hidden_list, train_acc_list, test_acc_list)

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def three_layer(inputSize=784, hidden1=0, hidden2=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = ThreeLayerNet(input_size=inputSize, hidden_size1=hidden1, hidden_size2=hidden2, output_size=10)

    # 하이퍼파라미터
    iters_num = 10000  # 반복 횟수 설정
    train_size = x_train.shape[0]
    batch_size = 100  # 미니배치 크기
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1 에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 오차역전파 방식(속도 빠름)

        # 계층 별로 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]

        # 학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    hidden_list = [hidden1, hidden2]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    # 그래프 그리기
    grape1(hidden_list, train_acc_list, test_acc_list)

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def four_layer(inputSize=784, hidden1=0, hidden2=0, hidden3=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = FourLayerNet(input_size=inputSize, hidden_size1=hidden1, hidden_size2=hidden2, hidden_size3=hidden3,
                           output_size=10)

    # 하이퍼파라미터
    iters_num = 10000  # 반복 횟수 설정
    train_size = x_train.shape[0]
    batch_size = 100  # 미니배치 크기
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1 에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 오차역전파 방식(속도 빠름)

        # 계층 별로 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
            network.params[key] -= learning_rate * grad[key]

        # 학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    hidden_list = [hidden1, hidden2, hidden3]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    # 그래프 그리기
    grape1(hidden_list, train_acc_list, test_acc_list)

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def five_layer(inputSize=784):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = FiveLayerNet(input_size=inputSize, hidden_size1=100, hidden_size2=75, hidden_size3=50, hidden_size4=25,
                           output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1 에폭당 반복 수
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산
        # grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 오차역전파 방식(속도 빠름)

        # 계층 별로 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'):
            network.params[key] -= learning_rate * grad[key]

        # 학습 경과 기록
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 1 에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            # print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    # 그래프 그리기
    grape1(train_acc_list, test_acc_list)

    hidden_list = [100, 75, 50, 25]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


# 각 레이어 시작 함수들
def two_train_start(input_size, start, end, skip):
    acc_avg_list = []

    for i in range(start, end, skip):
        acc_avg_list.append(two_layer(input_size, hidden1=i))

    grape2('two layer', acc_avg_list)


def three_train_start(input_size, start, end, skip):
    acc_avg_list = []

    for i in range(start, end, skip):
        for j in range(start, end, skip):
            acc_avg_list.append(three_layer(input_size, hidden1=i, hidden2=j))

    grape2('three layer', acc_avg_list)


def four_train_start(input_size, start, end, skip):
    acc_avg_list = []

    for i in range(start, end, skip):
        for j in range(start, end, skip):
            for k in range(start, end, skip):
                acc_avg_list.append(four_layer(input_size, hidden1=i, hidden2=j, hidden3=k))

    grape2('four layer', acc_avg_list)


# 정확도가 박살나 굳이 경우의 수를 구하지는 않았음.
def five_train_start(input_size):
    acc_av_list = [five_layer(input_size)]

    print('five layer 에서 테스트 기준 최대 평균 정확도를 보인 노드:', acc_av_list[0][0], '평균 트레이너 정확도:', round(acc_av_list[0][3], 5), '평균 테스트 정확도:', round(acc_av_list[0][4]), 5)


# 주어진 노드의 그래프를 그립니다.
def grape1(hidden_list, train_acc_list, test_acc_list):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.title(hidden_list)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    # plt.show()
    hidden = './img/'+'_'.join(str(e) for e in hidden_list)
    plt.savefig(fname=hidden)
    plt.clf()
    plt.close('all')


# 각 정확도의 평균치를 구하고, 최대 정확도를 보인 노드 경우의 수를 출력합니다.
def grape2(layer, acc_list):
    plt.clf()
    train_avg_list = []
    test_avg_list = []
    for i in range(0, len(acc_list)):
        train_avg_list.append(acc_list[i][3])
        test_avg_list.append(acc_list[i][4])

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_avg_list)) + 1
    plt.plot(x, train_avg_list, label='train avg')
    plt.plot(x, test_avg_list, label='test avg', linestyle='--')
    plt.title(layer)
    plt.xlabel("hidden node")
    plt.ylabel("accuracy")
    plt.xlim(0, (len(train_avg_list)+1))
    plt.ylim(min(min(np.min(train_avg_list), np.min(test_avg_list)) - 0.02, 0.9), 1)
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('./img/'+layer+'_acc')
    plt.clf()
    plt.close('all')

    count = 0
    for train_avg in train_avg_list:
        if train_avg == max(train_avg_list):
            print(layer, '에서 트레이너 기준 최대 평균 정확도를 보인 노드:', acc_list[count][0], ', 평균 트레이너 정확도:', round(train_avg, 5), ', 평균 테스트 정확도:', round(test_avg_list[count], 5))

        count += 1

    count = 0
    for test_avg in test_avg_list:
        if test_avg == max(test_avg_list):
            print(layer, '에서 테스트 기준 최대 평균 정확도를 보인 노드: ', acc_list[count][0], ', 평균 트레이너 정확도: ', round(train_avg_list[count], 5), ', 평균 테스트 정확도', round(test_avg, 5))
        count += 1
