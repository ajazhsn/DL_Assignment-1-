from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import sys
import os
import argparse
import json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


def _load_weights(model):
    """Try every possible source for weights, in order of reliability."""
    # 1. model_weights.py (embedded base64 - 100% reliable)
    for mwp in [os.path.join(_THIS_DIR, 'model_weights.py'),
                os.path.join(_ROOT_DIR, 'src', 'model_weights.py')]:
        if os.path.exists(mwp):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('mw', mwp)
                mw = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mw)
                model.set_weights(mw.get_weights())
                print(f"Weights loaded from {mwp}")
                return
            except Exception as e:
                print(f"model_weights.py error: {e}")

    # 2. JSON files
    for jp in [os.path.join(_THIS_DIR, 'best_model.json'),
               os.path.join(_ROOT_DIR, 'models', 'best_model.json'),
               os.path.join(_ROOT_DIR, 'src', 'best_model.json')]:
        if os.path.exists(jp):
            try:
                d = json.load(open(jp))
                model.set_weights({k: np.array(v) for k, v in d.items()})
                print(f"Weights loaded from {jp}")
                return
            except Exception as e:
                print(f"JSON error {jp}: {e}")

    # 3. NPY fallback
    for np_path in [os.path.join(_THIS_DIR, 'best_model.npy'),
                    os.path.join(_ROOT_DIR, 'models', 'best_model.npy')]:
        if os.path.exists(np_path):
            try:
                model.load(np_path)
                return
            except Exception as e:
                print(f"NPY error: {e}")


def main():
    args = parse_arguments()

    def _resolve(path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        c = os.path.join(_ROOT_DIR, path)
        return c if os.path.exists(c) else path

    # Load config - try multiple locations
    config_path = _resolve(args.config)
    for cp in [config_path,
               os.path.join(_THIS_DIR, 'best_config.json'),
               os.path.join(_ROOT_DIR, 'models', 'best_config.json')]:
        if os.path.exists(cp):
            config_path = cp
            break
    with open(config_path) as f:
        cfg = json.load(f)

    dataset = args.dataset or cfg.get('dataset', 'fashion_mnist')
    result = load_data(dataset)
    x_test, y_test = result[2], result[5]
    if hasattr(y_test, 'ndim') and y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)
    y_test = y_test.astype(int)

    hs = cfg.get('hidden_sizes', cfg.get('hidden_size', [128]))
    if isinstance(hs, int):
        hs = [hs]
    hs = [int(h) for h in hs]

    model = NeuralNetwork(input_size=int(x_test.shape[1]), hidden_sizes=hs,
                          output_size=10, activation=cfg.get('activation', 'relu'),
                          weight_init=cfg.get('weight_init', 'xavier'),
                          loss=cfg.get('loss', 'cross_entropy'))

    _load_weights(model)

    y_pred = model.predict(x_test).astype(int)

    if _SKLEARN:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(
            y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n========== Evaluation Results ==========")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(cm)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


if __name__ == '__main__':
    main()
