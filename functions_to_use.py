import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid_list = ['\sigma', '\sigma^{-1}', sigmoid]