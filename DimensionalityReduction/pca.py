from sklearn.decomposition import PCA


def pca(image, d):
    org_shape = image.shape
    pca_image = image.copy().reshape(org_shape[0] * org_shape[1], org_shape[2])
    pca_model = PCA(n_components=d)
    pca_image = pca_model.fit_transform(pca_image)
    pca_image = pca_image.reshape(org_shape[0], org_shape[1], d)
    return pca_image
