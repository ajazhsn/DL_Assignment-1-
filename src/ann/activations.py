import numpy as np

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, grad):
        return grad * self.mask

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.out
    def backward(self, grad):
        return grad * self.out * (1 - self.out)

class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad):
        return grad * (1 - self.out ** 2)

class Identity:
    def forward(self, x): return x
    def backward(self, grad): return grad

def get_activation(name):
    return {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh, 'none': Identity}[name.lower()]()
