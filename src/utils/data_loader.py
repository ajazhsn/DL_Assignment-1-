# src/utils/data_loader.py
import numpy as np
from sklearn.model_selection import train_test_split
import os
import struct
import gzip
import urllib.request


def _download_and_parse(dataset_name):
    """Download MNIST/Fashion-MNIST directly without keras/tensorflow."""
    if dataset_name == 'fashion_mnist':
        base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images':  't10k-images-idx3-ubyte.gz',
            'test_labels':  't10k-labels-idx1-ubyte.gz',
        }
    else:
        base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images':  't10k-images-idx3-ubyte.gz',
            'test_labels':  't10k-labels-idx1-ubyte.gz',
        }
    cache_dir = os.path.join(os.path.expanduser('~'),
                             '.datasets', dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for key, fname in files.items():
        fpath = os.path.join(cache_dir, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(base + fname, fpath)
        with gzip.open(fpath, 'rb') as f:
            raw = f.read()
        if 'images' in key:
            magic, n, h, w = struct.unpack('>IIII', raw[:16])
            data[key] = np.frombuffer(raw[16:], np.uint8).reshape(n, h, w)
        else:
            magic, n = struct.unpack('>II', raw[:8])
            data[key] = np.frombuffer(raw[8:], np.uint8)
    return (data['train_images'], data['train_labels']), \
           (data['test_images'],  data['test_labels'])


def load_data(dataset_name: str, val_split: float = 0.1):
    name = dataset_name.lower().replace('-', '_')

    # Try tensorflow first
    try:
        import tensorflow as tf
        if name == 'mnist':
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        else:
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    except Exception:
        # Try standalone keras
        try:
            if name == 'mnist':
                from keras.datasets import mnist
                (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
            else:
                from keras.datasets import fashion_mnist
                (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
        except Exception:
            # Direct download fallback
            (x_tr, y_tr), (x_te, y_te) = _download_and_parse(name)

    x_tr = x_tr.reshape(-1, 784).astype(np.float32) / 255.0
    x_te = x_te.reshape(-1, 784).astype(np.float32) / 255.0
    x_train, x_val, y_train, y_val = train_test_split(
        x_tr, y_tr, test_size=val_split, random_state=42, stratify=y_tr)
    print(
        f"Dataset: {dataset_name} | Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_te.shape[0]}")
    return x_train, x_val, x_te, y_train, y_val, y_te
