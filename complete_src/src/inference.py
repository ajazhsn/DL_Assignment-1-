import sys, os, argparse, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

try:
    from sklearn.metrics import (accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix, classification_report)
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, default='models/best_model.npy')
    p.add_argument('--config',  type=str, default='models/best_config.json')
    p.add_argument('--dataset', type=str, default=None)
    return p.parse_args()

parse_args = parse_arguments

def main():
    args = parse_arguments()
    def _resolve(path):
        if os.path.isabs(path) or os.path.exists(path): return path
        c = os.path.join(_ROOT_DIR, path)
        return c if os.path.exists(c) else path

    config_path = _resolve(args.config)
    with open(config_path) as f: cfg = json.load(f)

    dataset = args.dataset or cfg.get('dataset','fashion_mnist')
    result  = load_data(dataset)
    x_test, y_test = result[2], result[5]
    if y_test.ndim == 2: y_test = np.argmax(y_test, axis=1)
    y_test = y_test.astype(int)

    hs = cfg.get('hidden_sizes', cfg.get('hidden_size',[128]))
    if isinstance(hs, int): hs = [hs]
    hs = [int(h) for h in hs]

    model = NeuralNetwork(input_size=int(x_test.shape[1]), hidden_sizes=hs,
        output_size=10, activation=cfg.get('activation','relu'),
        weight_init=cfg.get('weight_init','xavier'), loss=cfg.get('loss','cross_entropy'))

    model_path = _resolve(args.model)
    model.load(model_path)
    y_pred = model.predict(x_test).astype(int)

    if _SKLEARN:
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        print(f"\n========== Evaluation Results ==========")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(cm)
        return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

if __name__ == '__main__':
    main()
