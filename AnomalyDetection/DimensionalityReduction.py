from sklearn.decomposition import NMF
import numpy as np
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


def mnf(image, max_iter, k):
    assert len(image.shape) == 3
    h, w, c = image.shape
    image = image.reshape((h * w, c))

    model = NMF(n_components=k, init='random', random_state=0, max_iter=max_iter)
    W = model.fit_transform(image)
    W = W.reshape((h, w, k))

    return W


def svd(image, k):
    assert len(image.shape) == 3, "Input image must be a 3D array"
    h, w, c = image.shape
    if k > c:
        raise ValueError("Number of components k cannot be greater than the number of channels")

    # Reshape the image from (h, w, c) to (h*w, c)
    image_flat = image.reshape((h * w, c))

    U, S, Vt = np.linalg.svd(image_flat, full_matrices=False)

    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])

    image_reduced = U_reduced @ S_reduced
    image_reduced = image_reduced.reshape((h, w, k))

    return image_reduced


def pca(image, k):
    org_shape = image.shape
    pca_image = image.copy().reshape(org_shape[0] * org_shape[1], org_shape[2])
    pca_model = PCA(n_components=k)
    pca_image = pca_model.fit_transform(pca_image)
    pca_image = pca_image.reshape(org_shape[0], org_shape[1], k)
    return pca_image
