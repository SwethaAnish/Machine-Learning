import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class weight_init:
    def __init__(self, func):
        self.input = np.random.randn(1,4)
        self.activation_dict = {'tanh':lambda x:np.tanh(x),'sigmoid':lambda x:1/(1+np.exp(-x)), 'relu': lambda x : np.maximum(0, x) }
        self.hidden_layer = [10] * 5 # [5 hidden layers with 10 neurons each]
        self.H_matrix = {}
        self.func = func
        self.comp_values(self.func)
    def comp_values(self, func):
        dict_ = {}
        for i in range(len(self.hidden_layer)):
            if i == 0:
                X = self.input  
            else:
                X = self.H_matrix[i - 1]
            self.fan_in = X.shape[1]
            self.fan_out = self.hidden_layer[i]
            W = eval('self.' + func)(self.fan_in, self.fan_out)['weight']
            activation = eval('self.' + func)(self.fan_in, self.fan_out)['activation']
            self.act_output = np.dot(X, W)
            self.act_output = self.activation_dict[activation](self.act_output)
            self.H_matrix[i] = self.act_output
            dict_[f"Hidden layer {i + 1}"] = list(self.H_matrix[i][0])
        df = pd.DataFrame(dict_)
        df.plot.line()
        plt.show()
    def zero(self, fan_in, fan_out):
        self.weight = np.zeros((self.fan_in, self.fan_out), dtype=int)
        self.activation = 'tanh'
        return {'weight': self.weight, 'activation':self.activation}
    def someValue_k(self):
        pass
    def random_small(self):
        pass
    def random_large(self):
        pass
    def xavier(self):
        pass
    def he_(self):
        pass

weight_init("zero")