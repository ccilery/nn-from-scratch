"""
    用numpy实现全连接神经网络, 用于mnist手写数字识别任务
"""
import numpy as np 

from utils.mnist_loader import load_data, load_batches

###################### 激活函数和导数 ##############################
def relu(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    flag = (z <= 0) # 需要修改为0的部分
    z[flag] = 0
    return z

def derivation_relu(z):
    flag = (z <= 0)
    z[flag] = 0
    z[~flag] = 1
    return z

def sigmoid(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    return 1 / (1 + np.exp(-z))

def derivation_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def derivation_tanh(z):
    return 1 - tanh(z)**2

def softmax(z):
    """
    Args:
        z: (batch_size, output_size)
    Returns:
        (batch_size, output_size)
    """
    max_row = np.max(z, axis=-1, keepdims=True)  # 每一个样本的所有分数中的最大值
    tmp = z - max_row
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=-1, keepdims=True)

def softmax_cross_entropy(logits, y):
    """
    Args:
        logits: (batch_size, output_size)， 网络的输出预测得分, 还没有进行 softmax概率化
        y: (batch_size, ) 每个样本的真实label
    return:
        a: (batch_size, output_size)
        loss: scalar
    """
    n = logits.shape[0]
    a = softmax(logits)
    scores = a[range(n), y]
    loss = -np.sum(np.log(scores)) / n 
    return a, loss

def derivation_softmax_cross_entropy(logits, y):
    """
    Args:
        logits: (batch_size, output_size)， 网络的输出预测得分, 还没有进行 softmax概率化
        y: (batch_size, ) 每个样本的真实label
    
    Return:
        \frac {\partial C}{\partial z^L}
        (batch_size, output_size)
    """
    n = logits.shape[0]
    a = softmax(logits)
    a[range(n), y] -= 1
    return a


class Network(object):
    """
    fully-connected neural network

    Attributions:
        sizes: list, 每个元素是每层的神经元的个数, 包括输入输出层
        num_layers: 神经网络的层数
        weights: list, 每个元素是一层神经网络的权重
        bias: list, 每个元素是一层神经网络的偏置
    """
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(i, j) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.bias = [np.random.randn(1, j) for j in self.sizes[1:]]


    def forward(self, x):
        """
        用于推理，前向传播时不进行softmax概率化
        x: (batch_size, input_size)
        """
        a = x
        for weight, bias in zip(self.weights[:-1], self.bias[:-1]):
            z = np.dot(a, weight) + bias
            a = relu(z) 
        # 在前向传播时不需要进行softmax概率化， 反向传播时才会用到
        logits = np.dot(a, self.weights[-1]) + self.bias[-1]
        return logits

    def backward(self, x, y):
        """
        Args:
            x: (batch_size, input_size)
            y: (batch_size, )
        returns:
            dws: list， 每个元素是每一层权重的梯度
            dbs: list, 每个元素是每一层偏置的梯度
        """
        # 存储每一层的损失函数对参数的梯度
        dws = [np.zeros((i, j)) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        dbs = [np.zeros((1, j)) for j in self.sizes[1:]]

        ################# 前向传播 ##############################
        # zs, _as存储前向传播过程中的中间变量z和a，供反向传播时使用
        zs = [] 
        _as = []

        a = x
        _as.append(a)
        for weight, bias in zip(self.weights[:-1], self.bias[:-1]):
            z = np.dot(a, weight) + bias
            zs.append(z)
            a = relu(z)
            _as.append(a)
        # 输出层
        logits = np.dot(a, self.weights[-1]) + self.bias[-1]
        zs.append(logits)
        a, loss = softmax_cross_entropy(logits, y)
        _as.append(a)

        ################# 反向传播 ##############################
        # 输出层误差
        dl = derivation_softmax_cross_entropy(logits, y)
        # batch的大小
        n = len(x)
        # 最后一层的梯度
        # 每个样本得的梯度求和、求平均
        dws[-1] = np.dot(_as[-2].T, dl) / n
        dbs[-1] = np.sum(dl, axis=0, keepdims=True) / n
        # 误差反向传播
        for i in range(2, self.num_layers):
            dl = np.dot(dl, self.weights[-i+1].T) * derivation_relu(zs[-i])
            dws[-i] = np.dot(_as[-i-1].T, dl) / n
            dbs[-i] = np.sum(dl, axis=0, keepdims=True) / n

        return loss, dws, dbs

    def train(self, training_data, validation_data, learning_rate, epochs, batch_size):
        for epoch in range(epochs):           
            ####################### 训练集进行训练 #############################
            x_batches, y_batches = load_batches(training_data, batch_size)

            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                loss, dws, dbs = self.backward(x, y)

                self.weights = [weight - learning_rate * dw for weight, dw in zip(self.weights, dws)]
                self.bias = [bias - learning_rate * db for bias, db in zip(self.bias, dbs)]

                if i % 100 == 0:
                    print("Epoch {}, batch {}, loss {}".format(epoch, i, loss))

            ######################## 验证集进行evaluate ##########################
            x_batches, y_batches = load_batches(validation_data, batch_size)

            corrects = 0
            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                logits = self.forward(x)
                correct = np.sum(np.argmax(logits, axis=-1) == y)
                corrects += correct
            
            print("Epoch {}, acc {}/{}={}".format(epoch, corrects, len(validation_data[0]), corrects/len(validation_data[0])))


def main():
    path = "data/mnist.pkl.gz"
    training_data, validation_data, test_data = load_data(path)
    model = Network([784, 30, 10])
    model.train(training_data, validation_data, 1, 50, 100)

if __name__ == "__main__":
    main()




