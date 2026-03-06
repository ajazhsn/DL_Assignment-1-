import numpy as np

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        self.y_true = np.array(y_true).astype(int)
        self.batch_size = logits.shape[0]
        eps = 1e-12
        return -np.log(self.probs[np.arange(self.batch_size), self.y_true] + eps).mean()

    def backward(self, *args, **kwargs):
        if not hasattr(self, 'probs'):
            return None
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.y_true] -= 1
        return grad / self.batch_size

class MSELoss:
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        n = logits.shape[0]
        self.y_true = np.array(y_true).astype(int)
        one_hot = np.eye(logits.shape[1])[self.y_true]
        self.diff = self.probs - one_hot
        self.batch_size = n
        return (self.diff ** 2).mean()

    def backward(self, *args, **kwargs):
        if not hasattr(self, 'probs'):
            return None
        dA = 2 * self.diff / (self.batch_size * self.diff.shape[1])
        return self.probs * (dA - (dA * self.probs).sum(axis=1, keepdims=True))

def get_loss(name):
    return {'cross_entropy': CrossEntropyLoss, 'mean_squared_error': MSELoss, 'mse': MSELoss}[name.lower()]()
