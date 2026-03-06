import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset_name: str, val_split: float = 0.1):
    name = dataset_name.lower().replace('-', '_')

    def _try_load(name):
        # Try tensorflow.keras first
        try:
            import tensorflow as tf
            if name == 'mnist':
                return tf.keras.datasets.mnist.load_data()
            return tf.keras.datasets.fashion_mnist.load_data()
        except Exception:
            pass
        # Try standalone keras
        try:
            if name == 'mnist':
                from keras.datasets import mnist; return mnist.load_data()
            from keras.datasets import fashion_mnist; return fashion_mnist.load_data()
        except Exception:
            pass
        raise RuntimeError(f"Cannot load {name}. Install tensorflow or keras.")

    (x_train_full, y_train_full), (x_test, y_test) = _try_load(name)
    x_train_full = x_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    x_test       = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_split, random_state=42, stratify=y_train_full)
    print(f"Dataset: {dataset_name} | Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    return x_train, x_val, x_test, y_train, y_val, y_test
