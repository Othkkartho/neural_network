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


def two_layer(hidden1=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 2~4 계층까지 데이터 생성해 저장. 4 계층 이후는 정확도 올라가는 속도가 너무 늦거나, 안올라 가기 때문에 이후는 안만듬. 2~4 까지만 하면 될 듯, 레이어 노드 수를 변경하려면
    # hidden_size를 변경 input_size를 변경한다면 dataset 파일 mnist_ass 코드의 img_size 변수도 변경시켜야 함. (변경 추천 수치, 392, 196)
    network = TwoLayerNet(input_size=784, hidden_size=hidden1, output_size=10)

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
            print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    # 그래프 그리기
    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_acc_list))
    # plt.plot(x, train_acc_list, label='train acc')
    # plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.ylim(0, 1.0)
    # plt.legend(loc='lower right')
    # plt.show()

    hidden_list = [hidden1, 0, 0]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def three_layer(hidden1=0, hidden2=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 2~4 계층까지 데이터 생성해 저장. 4 계층 이후는 정확도 올라가는 속도가 너무 늦거나, 안올라 가기 때문에 이후는 안만듬. 2~4 까지만 하면 될 듯, 레이어 노드 수를 변경하려면 hidden_size를 변경
    # input_size를 변경한다면 dataset 파일 mnist_ass 코드의 img_size 변수도 변경시켜야 함. (변경 추천 수치, 392, 196)
    network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)

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
            print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    # 그래프 그리기
    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_acc_list))
    # plt.plot(x, train_acc_list, label='train acc')
    # plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.ylim(0, 1.0)
    # plt.legend(loc='lower right')
    # plt.show()

    hidden_list = [hidden1, hidden2, 0]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def four_layer(hidden1=0, hidden2=0, hidden3=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 2~4 계층까지 데이터 생성해 저장. 4 계층 이후는 정확도 올라가는 속도가 너무 늦거나, 안올라 가기 때문에 이후는 안만듬. 2~4 까지만 하면 될 듯, 레이어 노드 수를 변경하려면 hidden_size를 변경
    # input_size를 변경한다면 dataset 파일 mnist_ass 코드의 img_size 변수도 변경시켜야 함. (변경 추천 수치, 392, 196)
    network = FourLayerNet(input_size=784, hidden_size1=100, hidden_size2=75, hidden_size3=50, output_size=10)

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
            print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    # 그래프 그리기
    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_acc_list))
    # plt.plot(x, train_acc_list, label='train acc')
    # plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.ylim(0, 1.0)
    # plt.legend(loc='lower right')
    # plt.show()

    hidden_list = [hidden1, hidden2, hidden3]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def five_layer(self=0):
    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 2~4 계층까지 데이터 생성해 저장. 4 계층 이후는 정확도 올라가는 속도가 너무 늦거나, 안올라 가기 때문에 이후는 안만듬. 2~4 까지만 하면 될 듯, 레이어 노드 수를 변경하려면 hidden_size를 변경
    # input_size를 변경한다면 dataset 파일 mnist_ass 코드의 img_size 변수도 변경시켜야 함. (변경 추천 수치, 392, 196)
    network = FiveLayerNet(input_size=784, hidden_size1=100, hidden_size2=75, hidden_size3=50, hidden_size4=25,
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
        # grad = network.numerical_gradient(x_batch, t_batch)   # 수치 미분 방식 <- 원래는 이거 써야함. 그런데 속도가 상당히 느림. 만약 하다가 열받으면 그냥 아래꺼 쓸것
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
            print("train_acc: ", train_acc, ", test_acc: ", test_acc)

    # 그래프 그리기
    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_acc_list))
    # plt.plot(x, train_acc_list, label='train acc')
    # plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.ylim(0, 1.0)
    # plt.legend(loc='lower right')
    # plt.show()

    hidden_list = [100, 75, 50, 25]
    train_avg = sum(train_acc_list[-10:]) / 10
    test_avg = sum(test_acc_list[-10:]) / 10

    return hidden_list, train_acc_list, test_acc_list, train_avg, test_avg


def two_train_start(start, end, skip):
    acc_avg_list = []

    for i in range(start, end, skip):
        acc_avg_list.append(two_layer(hidden1=i))

    train_avg_list = []
    test_avg_list = []
    for i in range(0, len(acc_avg_list)):
        train_avg_list.append(acc_avg_list[i][3])
        test_avg_list.append(acc_avg_list[i][4])

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_avg_list))+1
    plt.plot(x, train_avg_list, 'bo-', label='train avg')
    plt.plot(x, test_avg_list, 'bo--', label='test avg')
    plt.xlabel("hidden node")
    plt.ylabel("accuracy")
    plt.ylim(min(np.min(train_avg_list), np.min(test_avg_list))-0.02, 1)
    plt.legend(loc='lower right')
    plt.show()

    # 최대 최소 train acc와 test acc 를 hidden size와 함께 print됨.


def three_train_start(start, end, skip):
    acc_av_list = []

    for i in range(start, end, skip):
        for j in range(start, end, skip):
            acc_av_list.append(three_layer(hidden1=i, hidden2=j))

    print(acc_av_list)


def four_train_start(start, end, skip):
    acc_av_list = []

    for i in range(start, end, skip):
        for j in range(start, end, skip):
            for k in range(start, end, skip):
                acc_av_list.append(four_layer(hidden1=i, hidden2=j, hidden3=k))

    print(acc_av_list)


def five_train_start():
    acc_av_list = [five_layer()]

    print(acc_av_list)
