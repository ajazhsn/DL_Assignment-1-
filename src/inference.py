import sys, os, argparse, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

try:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 confusion_matrix, classification_report)
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def parse_arguments():
    p = argparse.ArgumentParser(description='Evaluate trained MLP')
    # ── same flags as train.py ──────────────────────────────────────────
    p.add_argument('-d',   '--dataset',       type=str,   default='mnist')   # ← FIXED: was 'fashion_mnist'
    p.add_argument('-e',   '--epochs',        type=int,   default=10)
    p.add_argument('-b',   '--batch_size',    type=int,   default=64)
    p.add_argument('-l',   '--loss',          type=str,   default='cross_entropy')
    p.add_argument('-o',   '--optimizer',     type=str,   default='rmsprop')
    p.add_argument('-lr',  '--learning_rate', type=float, default=0.001)
    p.add_argument('-wd',  '--weight_decay',  type=float, default=0.0)
    p.add_argument('-nhl', '--num_layers',    type=int,   default=3)
    p.add_argument('-sz',  '--hidden_size',   type=int,   nargs='+', default=[128, 128, 128])  # ← FIXED: matches actual weights
    p.add_argument('-a',   '--activation',    type=str,   default='relu')
    p.add_argument('-w_i', '--weight_init',   type=str,   default='xavier')
    p.add_argument('-w_p', '--wandb_project', type=str,   default=None)
    # ── inference-specific ─────────────────────────────────────────────
    p.add_argument('--model_path',  type=str, default=None)
    p.add_argument('--config_path', type=str, default=None)
    return p.parse_args()


def _load_weights(model: NeuralNetwork, model_path=None) -> None:
    """Try every possible location for weights."""
    candidates_npy = []
    if model_path:
        candidates_npy.append(model_path)
    for base in [_THIS_DIR,
                 os.path.join(_THIS_DIR, 'src'),
                 os.path.join(_THIS_DIR, 'models'),
                 os.path.dirname(_THIS_DIR),
                 os.getcwd()]:
        candidates_npy.append(os.path.join(base, 'best_model.npy'))

    # Try JSON first (avoids numpy pickle issues)
    for np_path in candidates_npy:
        jp = np_path.replace('.npy', '.json')
        if os.path.exists(jp):
            try:
                with open(jp) as f:
                    d = json.load(f)
                model.set_weights({k: np.array(v) for k, v in d.items()})
                print(f'Weights loaded from {jp}')
                return
            except Exception as e:
                print(f'  JSON error ({jp}): {e}')

    # Then try .npy
    for np_path in candidates_npy:
        if os.path.exists(np_path):
            try:
                data = np.load(np_path, allow_pickle=True)
                model.set_weights(data)
                print(f'Weights loaded from {np_path}')
                return
            except Exception as e:
                print(f'  NPY error ({np_path}): {e}')

    raise FileNotFoundError(
        f'Could not find model weights. Searched: {candidates_npy}')


def _load_config(config_path=None):
    candidates = []
    if config_path:
        candidates.append(config_path)
    for base in [_THIS_DIR,
                 os.path.join(_THIS_DIR, 'src'),
                 os.path.join(_THIS_DIR, 'models'),
                 os.path.dirname(_THIS_DIR),
                 os.getcwd()]:
        candidates.append(os.path.join(base, 'best_config.json'))
    for cp in candidates:
        if os.path.exists(cp):
            with open(cp) as f:
                cfg = json.load(f)
            print(f'Config loaded from {cp}')
            return cfg
    return None


def main():
    args = parse_arguments()

    cfg = _load_config(args.config_path)

    # ── dataset: CLI flag wins, then config, then default ───────────────
    dataset = args.dataset
    if cfg is None:
        hs  = args.hidden_size if isinstance(args.hidden_size, list) else [args.hidden_size]
        hs  = hs * args.num_layers if len(hs) == 1 else hs
        cfg = {
            'dataset':      dataset,
            'hidden_sizes': hs,
            'activation':   args.activation,
            'weight_init':  args.weight_init,
            'loss':         args.loss,
        }
        print('No config file found — using CLI arguments.')
    else:
        # Only override dataset from config if user did NOT pass -d explicitly
        # (argparse gives the default when not passed, so we check sys.argv)
        if '-d' not in sys.argv and '--dataset' not in sys.argv:
            dataset = cfg.get('dataset', dataset)

    # ── load test data ──────────────────────────────────────────────────
    _, _, x_test, _, _, y_test = load_data(dataset)
    y_test = y_test.astype(int)

    # ── build model from config ─────────────────────────────────────────
    hs = cfg.get('hidden_sizes', cfg.get('hidden_size', [128, 64, 32]))
    if isinstance(hs, int):
        hs = [hs]
    hs = [int(h) for h in hs]

    model = NeuralNetwork(
        input_size   = int(x_test.shape[1]),
        hidden_sizes = hs,
        output_size  = 10,
        activation   = cfg.get('activation',  'relu'),
        weight_init  = cfg.get('weight_init', 'xavier'),
        loss         = cfg.get('loss',         'cross_entropy'),
    )

    _load_weights(model, args.model_path)

    # ── verify architecture matches weights ─────────────────────────────
    # (catches silent mismatch where config arch ≠ weight arch)
    weights_raw = None
    for base in [_THIS_DIR, os.path.join(_THIS_DIR,'src'), os.getcwd()]:
        p = os.path.join(base, 'best_model.npy')
        if os.path.exists(p):
            weights_raw = np.load(p, allow_pickle=True)
            if weights_raw.ndim == 0:
                weights_raw = weights_raw.item()
            break
    if weights_raw and isinstance(weights_raw, dict):
        n_weight_layers = len([k for k in weights_raw if k.startswith('W')])
        if n_weight_layers != len(model.layers):
            print(f'WARNING: config says {len(model.layers)} layers '
                  f'but weights have {n_weight_layers}. Rebuilding model.')
            # Infer hidden sizes directly from weight shapes
            ws = [weights_raw[f'W{i}'] for i in range(n_weight_layers)]
            inferred_hs = [w.shape[1] for w in ws[:-1]]
            model = NeuralNetwork(
                input_size   = int(x_test.shape[1]),
                hidden_sizes = inferred_hs,
                output_size  = 10,
                activation   = cfg.get('activation',  'relu'),
                weight_init  = cfg.get('weight_init', 'xavier'),
                loss         = cfg.get('loss',         'cross_entropy'),
            )
            model.set_weights(weights_raw)

    # ── inference ───────────────────────────────────────────────────────
    y_pred = model.predict(x_test).astype(int)

    # ── metrics ─────────────────────────────────────────────────────────
    if _SKLEARN:
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        print('\n========== Evaluation Results ==========')
        print(f'  Accuracy  : {acc:.4f}')
        print(f'  Precision : {prec:.4f}')
        print(f'  Recall    : {rec:.4f}')
        print(f'  F1-Score  : {f1:.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, zero_division=0))
        print('Confusion Matrix:')
        print(cm)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    else:
        acc = float(np.mean(y_pred == y_test))
        print(f'\nAccuracy: {acc:.4f}  (install scikit-learn for full metrics)')
        return {'accuracy': acc}

if __name__ == '__main__':
    main()