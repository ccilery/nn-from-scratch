"""
mnist数据集的加载
"""

import pickle
import gzip
import random
import numpy as np

def load_data(path):
    """
        Args:
            path: mnist数据集路径
        Returns:
            training_data: tuple
                training_data[0]: 输入数据, (num_samples, input_size) 即(50000, 784)
                training_data[1]: 标签, (num_samples, ) 即(50000, )
    """
    f = gzip.open(path)
    training_data, val_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, val_data, test_data

def load_batches(data, batch_size):
    """
        对数据洗牌，并分成一个个batch
        Args:
            data: tuple
                data[0]: 输入数据, (num_samples, input_size)
                data[1]: 标签, (num_samples, )
            batch_size: 
        Returns:
            batches_x: list
                batches_x[0]: (batch_size, input_size)
            batches_y: list
                batches_y[0]: (batch_size, )
    """
    n = len(data[0])
    # 对数据进行洗牌
    shuffle_idx = random.sample(range(n), n)
    X = data[0][shuffle_idx]
    Y = data[1][shuffle_idx]

    batches_x = [X[i: i+batch_size] for i in range(0, n, batch_size)]
    batches_y = [Y[i: i+batch_size] for i in range(0, n, batch_size)]

    return batches_x, batches_y
   
 

def main():
    path = "data/mnist.pkl.gz"
    training_data, val_data, test_data = load_data(path)
    load_batches(training_data, 100)

if __name__ == "__main__":
    main()
