# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet_ass import SimpleConvNet
from three_convnet_ass import ThreeConvNet
from common.trainer_ass import Trainer

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20
for pad_num in range(0, 10, 2):
    for filter_num in range(10, 51, 10):
        network = SimpleConvNet(input_dim=(1, 28, 28),
                                conv_param={'filter_num': filter_num, 'filter_size': 5, 'pad': pad_num, 'stride': 1},
                                hidden_size=100, output_size=10, weight_init_std=0.01)

        # network = ThreeConvNet(input_dim=(1, 28, 28),
        #                         conv_param={'filter_num': filter_num, 'filter_size': 5, 'pad': pad_num, 'stride': 1},
        #                         hidden_size1=100, hidden_size2=95, output_size=10, weight_init_std=0.01)

        trainer = Trainer(network, x_train, t_train, x_test, t_test,
                          epochs=max_epochs, mini_batch_size=100,
                          optimizer='Adam', optimizer_param={'lr': 0.001},
                          evaluate_sample_num_per_epoch=1000)
        trainer.train(filter_num, pad_num)

        # 매개변수 보존
        network.save_params("params.pkl")
        print("Saved Network Parameters!")

        # 그래프 그리기
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(max_epochs)
        plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
        plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.title(str(filter_num) + " - " + str(pad_num) + " - SimpleConvNet")
        # plt.show()
        hidden = './img/' + str(pad_num) + "_" + str(filter_num) + "_" + "SimpleConvNet"
        plt.savefig(fname=hidden)
        plt.clf()
        plt.close('all')
