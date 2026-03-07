from utils.data_loader import load_data
from ann.optimizers import get_optimizer
from ann.neural_network import NeuralNetwork
import numpy as np
import json
import sys
import os
import base64
import pickle
sys.path.insert(0, 'src')
for k in list(sys.modules.keys()):
    if 'ann' in k or 'utils' in k:
        del sys.modules[k]


print('Loading data...')
x_train, x_val, x_test, y_train, y_val, y_test = load_data('mnist')

model = NeuralNetwork(784, [128, 128, 128], 10, activation='relu',
                      weight_init='xavier', loss='cross_entropy')
opt = get_optimizer('adam', lr=0.001)
best_acc = 0
best_w = None

for epoch in range(1, 16):
    idx = np.random.permutation(len(x_train))
    xtr, ytr = x_train[idx], y_train[idx]
    for i in range(0, len(xtr), 32):
        xb, yb = xtr[i:i+32], ytr[i:i+32]
        model.compute_loss(model.forward(xb), yb)
        model.backward()
        for l in model.layers:
            opt.update(l)
    acc = np.mean(model.predict(x_val) == y_val)
    print(f'Epoch {epoch}: val_acc={acc:.4f}')
    if acc > best_acc:
        best_acc = acc
        best_w = model.get_weights()

model.set_weights(best_w)
print(f'Best val_acc={best_acc:.4f}')

# Save as model_weights.py (embedded base64 - git corruption proof)
wb = base64.b64encode(pickle.dumps(best_w)).decode()
py_content = 'import numpy as _np, base64 as _b64, pickle as _pk\n'
py_content += '_W = "' + wb + '"\n'
py_content += 'def get_weights():\n'
py_content += '    return _pk.loads(_b64.b64decode(_W))\n'
open('src/model_weights.py', 'w').write(py_content)
print('Saved src/model_weights.py')

# Save JSON
jw = {k: v.tolist() for k, v in best_w.items()}
json.dump(jw, open('src/best_model.json', 'w'))
json.dump(jw, open('models/best_model.json', 'w'))
print('Saved JSON files')

# Save npy
np.save('src/best_model.npy', best_w)
np.save('models/best_model.npy', best_w)
print('Saved NPY files')

# Save config
cfg = {
    "dataset": "mnist",
    "hidden_sizes": [128, 128, 128],
    "hidden_size": [128, 128, 128],
    "num_layers": 3,
    "activation": "relu",
    "weight_init": "xavier",
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "best_val_acc": float(best_acc)
}
json.dump(cfg, open('src/best_config.json', 'w'), indent=2)
json.dump(cfg, open('models/best_config.json', 'w'), indent=2)
print(f'All done! best_val_acc={best_acc:.4f}')
