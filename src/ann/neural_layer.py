import numpy as np
from .activations import get_activation, Identity

class NeuralLayer:
    def __init__(self, input_size, output_size, activation, weight_init='xavier'):
        self.activation = activation
        if weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'random':
            self.W = np.random.randn(input_size, output_size) * 0.01
        else:
            self.W = np.zeros((input_size, output_size))
        self.b = np.zeros((1, output_size))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.optimizer_state = {}

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.activation.forward(self.z)

    def backward(self, grad_output):
        grad_z = self.activation.backward(grad_output)
        self.grad_W = self.x.T @ grad_z
        self.grad_b = grad_z.sum(axis=0, keepdims=True)
        return grad_z @ self.W.T
