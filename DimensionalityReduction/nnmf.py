import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def single_mnf(image, max_iter, k):
    model = NMF(n_components=k, init='random', random_state=0, max_iter=max_iter)
    W = model.fit_transform(image)
    H = model.components_
    V = W @ H
    error = np.sum((V - image)**2)
    return W, H, error


def sklearn_nmf(image, max_iter=1000, k=None):
    assert len(image.shape) == 3
    h, w, c = image.shape
    image = image.reshape((h*w, c))

    if k:
        W, H, err = single_mnf(image=image, max_iter=max_iter, k=k)
        W = W.reshape((h, w, k))
        return W
    else:
        best = {'err': np.inf, 'k': 2, 'W': None}
        k = 3
        losses = []
        while k < c:
            W, H, err = single_mnf(image=image, max_iter=max_iter, k=k)
            print(f"iter {k - 2} (k = {k}): loss = {err}")
            losses.append(err)
            if losses[-1] < best['err']:
                best = {'err': losses[-1], 'k': k, 'W': W}
            k += 1

        print(f'k = {best["k"]}')
        return best['W'].reshape((h, w, best['k']))
