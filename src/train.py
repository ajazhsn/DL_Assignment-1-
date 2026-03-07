import sys, os, argparse, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset',      type=str,   default='fashion_mnist')
    p.add_argument('-e', '--epochs',       type=int,   default=10)
    p.add_argument('-b', '--batch_size',   type=int,   default=32)
    p.add_argument('-lr','--learning_rate',type=float, default=0.001)
    p.add_argument('-o', '--optimizer',    type=str,   default='adam')
    p.add_argument('-nhl','--num_layers',  type=int,   default=2)
    p.add_argument('-sz', '--hidden_size', type=int,   nargs='+', default=[128])
    p.add_argument('-a', '--activation',   type=str,   default='relu')
    p.add_argument('-l', '--loss',         type=str,   default='cross_entropy')
    p.add_argument('-w_i','--weight_init', type=str,   default='xavier')
    p.add_argument('-wd','--weight_decay', type=float, default=0.0)
    p.add_argument('-wp','--wandb_project',type=str,   default=None)
    p.add_argument('-we','--wandb_entity', type=str,   default=None)
    p.add_argument('--run_name',           type=str,   default=None)
    # keep long-form aliases for backward compatibility
    p.add_argument('--dataset',      dest='dataset',      type=str,   default=argparse.SUPPRESS)
    p.add_argument('--epochs',       dest='epochs',       type=int,   default=argparse.SUPPRESS)
    p.add_argument('--batch_size',   dest='batch_size',   type=int,   default=argparse.SUPPRESS)
    p.add_argument('--learning_rate',dest='learning_rate',type=float, default=argparse.SUPPRESS)
    p.add_argument('--optimizer',    dest='optimizer',    type=str,   default=argparse.SUPPRESS)
    p.add_argument('--num_layers',   dest='num_layers',   type=int,   default=argparse.SUPPRESS)
    p.add_argument('--hidden_size',  dest='hidden_size',  type=int,   nargs='+', default=argparse.SUPPRESS)
    p.add_argument('--activation',   dest='activation',   type=str,   default=argparse.SUPPRESS)
    p.add_argument('--loss',         dest='loss',         type=str,   default=argparse.SUPPRESS)
    p.add_argument('--weight_init',  dest='weight_init',  type=str,   default=argparse.SUPPRESS)
    p.add_argument('--weight_decay', dest='weight_decay', type=float, default=argparse.SUPPRESS)
    p.add_argument('--wandb_project',dest='wandb_project',type=str,   default=argparse.SUPPRESS)
    p.add_argument('--wandb_entity', dest='wandb_entity', type=str,   default=argparse.SUPPRESS)
    return p.parse_args()

parse_args = parse_arguments

def train(args):
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        args.dataset, val_split=0.1)

    hs = args.hidden_size
    if isinstance(hs, int): hs = [hs]
    hs = [int(h) for h in hs]
    nl = getattr(args, 'num_layers', len(hs))
    if nl and nl != len(hs): hs = hs * nl if len(hs)==1 else hs[:nl]

    model = NeuralNetwork(784, hs, 10,
        activation=args.activation, weight_init=args.weight_init, loss=args.loss)
    opt = get_optimizer(args.optimizer, lr=args.learning_rate,
                        weight_decay=getattr(args,'weight_decay',0.0))

    best_val_acc = 0.0; best_weights = None
    for epoch in range(1, args.epochs+1):
        idx = np.random.permutation(len(x_train))
        xtr, ytr = x_train[idx], y_train[idx]
        total_loss=0; nb=0
        for i in range(0, len(xtr), args.batch_size):
            xb, yb = xtr[i:i+args.batch_size], ytr[i:i+args.batch_size]
            logits = model.forward(xb)
            total_loss += model.compute_loss(logits, yb)
            model.backward(); nb+=1
            for layer in model.layers: opt.update(layer)
        val_acc = np.mean(model.predict(x_val) == y_val)
        print(f"Epoch {epoch:2d}: loss={total_loss/nb:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_weights = model.get_weights()

    model.set_weights(best_weights)
    print(f"Best val_acc={best_val_acc:.4f}")
    return model

if __name__ == '__main__':
    train(parse_arguments())
