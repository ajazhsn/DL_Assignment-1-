import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr=lr; self.momentum=momentum; self.weight_decay=weight_decay; self.v={}
    def update(self, layer):
        lid=id(layer)
        if lid not in self.v: self.v[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b
        self.v[lid]['W'] = self.momentum*self.v[lid]['W'] - self.lr*gW
        self.v[lid]['b'] = self.momentum*self.v[lid]['b'] - self.lr*gb
        layer.W += self.v[lid]['W']; layer.b += self.v[lid]['b']

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps; self.wd=weight_decay
        self.m={}; self.v={}; self.t=0
    def update(self, layer):
        self.t+=1; lid=id(layer)
        if lid not in self.m:
            self.m[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
            self.v[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
        for p,g in [('W', layer.grad_W + self.wd*layer.W), ('b', layer.grad_b)]:
            self.m[lid][p]=self.b1*self.m[lid][p]+(1-self.b1)*g
            self.v[lid][p]=self.b2*self.v[lid][p]+(1-self.b2)*g**2
            mh=self.m[lid][p]/(1-self.b1**self.t)
            vh=self.v[lid][p]/(1-self.b2**self.t)
            if p=='W': layer.W -= self.lr*mh/(np.sqrt(vh)+self.eps)
            else:      layer.b -= self.lr*mh/(np.sqrt(vh)+self.eps)

class NAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps; self.wd=weight_decay
        self.m={}; self.v={}; self.t=0
    def update(self, layer):
        self.t+=1; lid=id(layer)
        if lid not in self.m:
            self.m[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
            self.v[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
        for p,g in [('W', layer.grad_W + self.wd*layer.W), ('b', layer.grad_b)]:
            self.m[lid][p]=self.b1*self.m[lid][p]+(1-self.b1)*g
            self.v[lid][p]=self.b2*self.v[lid][p]+(1-self.b2)*g**2
            mh=self.m[lid][p]/(1-self.b1**self.t)
            vh=self.v[lid][p]/(1-self.b2**self.t)
            update = (self.b1*mh + (1-self.b1)*g/(1-self.b1**self.t)) / (np.sqrt(vh)+self.eps)
            if p=='W': layer.W -= self.lr*update
            else:      layer.b -= self.lr*update

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr=lr; self.beta=beta; self.eps=eps; self.wd=weight_decay; self.v={}
    def update(self, layer):
        lid=id(layer)
        if lid not in self.v: self.v[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
        for p,g in [('W', layer.grad_W+self.wd*layer.W),('b',layer.grad_b)]:
            self.v[lid][p]=self.beta*self.v[lid][p]+(1-self.beta)*g**2
            if p=='W': layer.W -= self.lr*g/(np.sqrt(self.v[lid][p])+self.eps)
            else:      layer.b -= self.lr*g/(np.sqrt(self.v[lid][p])+self.eps)

def get_optimizer(name, lr=0.001, weight_decay=0.0, **kwargs):
    name = name.lower()
    opts = {'sgd': SGD, 'momentum': SGD, 'nesterov': SGD,
            'adam': Adam, 'nadam': NAdam, 'rmsprop': RMSProp}
    cls = opts.get(name, Adam)
    if name in ('momentum', 'nesterov'):
        return cls(lr=lr, momentum=0.9, weight_decay=weight_decay)
    return cls(lr=lr, weight_decay=weight_decay)
