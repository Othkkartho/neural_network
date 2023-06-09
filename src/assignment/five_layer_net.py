# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class FiveLayerNet:

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size1), 'b1': np.zeros(hidden_size1),
                       'W2': weight_init_std * np.random.randn(hidden_size1, hidden_size2), 'b2': np.zeros(hidden_size2),
                       'W3': weight_init_std * np.random.randn(hidden_size2, hidden_size3), 'b3': np.zeros(hidden_size3),
                       'W4': weight_init_std * np.random.randn(hidden_size3, hidden_size4), 'b4': np.zeros(hidden_size4),
                       'W5': weight_init_std * np.random.randn(hidden_size4, output_size), 'b5': np.zeros(output_size)}

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):  # 예측을 수행한다.
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):  # 손실값을 구한다.
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):  # 정확도를 구한다.
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):  # 가중치 매개변수의 기울기를 구한다.
        loss_W = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradient(loss_W, self.params['W1']), 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']), 'b2': numerical_gradient(loss_W, self.params['b2']),
                 'W3': numerical_gradient(loss_W, self.params['W3']), 'b3': numerical_gradient(loss_W, self.params['b3']),
                 'W4': numerical_gradient(loss_W, self.params['W4']), 'b4': numerical_gradient(loss_W, self.params['b4']),
                 'W5': numerical_gradient(loss_W, self.params['W5']), 'b5': numerical_gradient(loss_W, self.params['b5'])}

        return grads

    def gradient(self, x, t):           # 가중치 매개변수의 기울기를 구한다.
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW, 'b2': self.layers['Affine2'].db,
                 'W3': self.layers['Affine3'].dW, 'b3': self.layers['Affine3'].db,
                 'W4': self.layers['Affine4'].dW, 'b4': self.layers['Affine4'].db,
                 'W5': self.layers['Affine5'].dW, 'b5': self.layers['Affine5'].db}

        return grads
