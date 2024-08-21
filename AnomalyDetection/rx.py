import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve


def tyler_covariance_estimator(X, tol=1e-6, max_iter=100):
    # Need to be done!
    assert len(X.shape) == 2
    n, f = X.shape
    cov_matrix = np.cov(X, rowvar=False)
    return cov_matrix


def GRX(image, M=False):
    """
    :param M: is normal
    :param image: hyperspectral image / multi-dimension cube
    :return: rx score by global computation
    """
    height, width, bands = image.shape
    reshaped_array = image.reshape((height * width, -1))

    mean_vector = np.mean(reshaped_array, axis=0)
    covariance_matrix = np.cov(reshaped_array, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)

    rx_score = np.zeros((height, width))

    for i in tqdm(range(height)):
        for j in range(width):
            diff = image[i, j] - mean_vector
            rx_score[i, j] = (diff / np.linalg.norm(
                diff)).T @ inv_covariance_matrix @ diff if M else diff.T @ inv_covariance_matrix @ diff

    return rx_score


def GRXM(image):
    """
    :param image: hyperspectral image / multi-dimension cube
    :return: Normalized GRX
    """
    return GRX(image, M=True)


def LRX(image, guard_band_size=9, local_bg_size=23, local_cov=True):
    """
    :param image: hyperspectral image / multi-dimension cube
    :param guard_band_size: inner window size (estimated target size)
    :param local_bg_size: outer window size
    :param local_cov: compute the covariance matrix locally or globally
    :return: rx score of every pixel by outer window excludes inner window
    """
    guard_band_size = (guard_band_size, guard_band_size) if isinstance(guard_band_size, int) else guard_band_size
    local_bg_size = (local_bg_size, local_bg_size) if isinstance(local_bg_size, int) else local_bg_size

    assert guard_band_size[0] % 2 == 1 and guard_band_size[1] % 2 == 1 and local_bg_size[0] % 2 == 1 and local_bg_size[
        1] % 2 == 1

    height, width, bands = image.shape
    rx_scores = np.zeros((height, width))

    if local_cov:
        # Half sizes for the windows
        guard_band_half_height = guard_band_size[0] // 2
        guard_band_half_width = guard_band_size[1] // 2
        local_bg_half_height = local_bg_size[0] // 2
        local_bg_half_width = local_bg_size[1] // 2

        for i in tqdm(range(height)):
            for j in range(width):
                # Define the bounds for the outer window
                local_bg_top = max(0, i - local_bg_half_height)
                local_bg_bottom = min(height, i + local_bg_half_height + 1)
                local_bg_left = max(0, j - local_bg_half_width)
                local_bg_right = min(width, j + local_bg_half_width + 1)

                # Extract the outer window
                local_bg = image[local_bg_top:local_bg_bottom, local_bg_left:local_bg_right, :]
                mask = np.ones(local_bg.shape[:2], dtype=bool)

                # Define the bounds for the inner window
                guard_band_top = max(0, i - guard_band_half_height) - local_bg_top
                guard_band_bottom = min(height, i + guard_band_half_height + 1) - local_bg_top
                guard_band_left = max(0, j - guard_band_half_width) - local_bg_left
                guard_band_right = min(width, j + guard_band_half_width + 1) - local_bg_left

                mask[guard_band_top:guard_band_bottom, guard_band_left:guard_band_right] = False

                local_bg = local_bg[mask]
                mean_vector = np.mean(local_bg, axis=0)
                cov_matrix = np.cov(local_bg, rowvar=False)

                diff = image[i, j, :] - mean_vector

                try:
                    rx_score = diff.T @ np.linalg.inv(cov_matrix) @ diff
                except np.linalg.LinAlgError:
                    rx_score = np.inf

                rx_scores[i, j] = rx_score
    else:
        mean_matrix = np.zeros((height, width))
        mean_matrix_squared = np.zeros((height, width))

        squared_image = image ** 2

        kernel = np.ones((*local_bg_size, bands))

        center_y, center_x = local_bg_size[0] // 2, local_bg_size[1] // 2
        guard_start_y = center_y - guard_band_size[0] // 2
        guard_end_y = center_y + guard_band_size[0] // 2 + 1
        guard_start_x = center_x - guard_band_size[1] // 2
        guard_end_x = center_x + guard_band_size[1] // 2 + 1

        kernel[guard_start_y:guard_end_y, guard_start_x:guard_end_x, :] = 0

        kernel /= np.sum(kernel)

        for b in range(bands):
            mean_matrix += convolve(image[:, :, b], kernel[:, :, b])
            mean_matrix_squared += convolve(squared_image[:, :, b], kernel[:, :, b])

        reg = 1e-6
        std_matrix = np.sqrt(mean_matrix_squared - mean_matrix**2) + reg

        diff = (image - mean_matrix[:, :, np.newaxis]) / std_matrix[:, :, np.newaxis]

        cov_matrix = tyler_covariance_estimator(image.reshape(height * width, -1))#np.cov(image.reshape(height * width, -1), rowvar=False)
        cov_matrix_inv = np.linalg.inv(cov_matrix)

        rx_scores = np.einsum('ijk,kl,ijl->ij', diff, cov_matrix_inv, diff)

    return rx_scores


def RX_UTD(image):
    """
    :param image: hyperspectral image / multi-dimension cube
    :return: rx score by global computation + UTD
    """
    height, width, bands = image.shape
    reshaped_array = image.reshape((height * width, -1))

    mean_vector = np.mean(reshaped_array, axis=0)
    covariance_matrix = np.cov(reshaped_array, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)

    rx_score = np.zeros((height, width))

    for i in tqdm(range(height)):
        for j in range(width):
            diff = image[i, j, :] - mean_vector
            rx_score[i, j] = (image[i, j, :] - 1) @ inv_covariance_matrix @ diff.T

    return rx_score


def DWRX(image, inner_window_size=(5, 5), outer_window_size=(15, 15)):
    """
    :param image: hyperspectral image / multi-dimension cube
    :param inner_window_size: the size of the first window in rx score
    :param outer_window_size: the size of the second window in rx score
    :return: the rx score with the different between the first window to second window
    """
    height, width, bands = image.shape
    rx_scores = np.zeros((height, width))

    # Half sizes for the windows
    inner_half_height = inner_window_size[0] // 2
    inner_half_width = inner_window_size[1] // 2
    outer_half_height = outer_window_size[0] // 2
    outer_half_width = outer_window_size[1] // 2

    for i in tqdm(range(height)):
        for j in range(width):
            # Define the bounds for the outer window
            outer_top = max(0, i - outer_half_height)
            outer_bottom = min(height, i + outer_half_height + 1)
            outer_left = max(0, j - outer_half_width)
            outer_right = min(width, j + outer_half_width + 1)

            # Extract the outer window
            outer_window = image[outer_top:outer_bottom, outer_left:outer_right, :]

            # Calculate the mean and covariance of the outer window
            mean_outer = np.mean(outer_window, axis=(0, 1))
            cov_outer = np.cov(outer_window.reshape(-1, bands), rowvar=False)

            # Define the bounds for the inner window
            inner_top = max(0, i - inner_half_height)
            inner_bottom = min(height, i + inner_half_height + 1)
            inner_left = max(0, j - inner_half_width)
            inner_right = min(width, j + inner_half_width + 1)

            # Extract the inner window
            inner_window = image[inner_top:inner_bottom, inner_left:inner_right, :]

            # Calculate the mean of the inner window
            mean_inner = np.mean(inner_window, axis=(0, 1))

            # Calculate the LRX score
            diff = mean_inner - mean_outer
            try:
                lrx_score = diff.T @ np.linalg.inv(cov_outer) @ diff
            except np.linalg.LinAlgError:
                lrx_score = np.inf  # Handle singular covariance matrix

            rx_scores[i, j] = lrx_score

    return rx_scores


def SSRX(image, num_vectors=1):
    """
    :param image: hyperspectral image / multi-dimension cube
    :param num_vectors: number of eigen vectors to take (by highest eigen values)
    :return: rx score but with eigen vectors projection
    """
    height, width, bands = image.shape

    assert num_vectors < bands

    reshaped_array = image.reshape((height * width, -1))

    mean_vector = np.mean(reshaped_array, axis=0)
    covariance_matrix = np.cov(reshaped_array, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    W = eigenvectors[:, eigenvalues.argsort()[::-1][:num_vectors]].T
    W_pseudo_inv = np.linalg.pinv(W)

    rx_scores = np.zeros((height, width))

    for i in tqdm(range(height)):
        for j in range(width):
            diff = image[i, j, :] - mean_vector
            I = np.eye(bands)
            rx_scores[i, j] = diff.T @ (I - (W @ W_pseudo_inv)) @ diff

    return -rx_scores


def KRX(image, c=3):
    """
    :param image: hyperspectral image / multi-dimension cube
    :param c: sigma for gaussian kernel
    :return: rx score but after projection to kernel extra-dimensions (RBF kernel)
    """
    def gram_matrix_rbf(X, sigma=1.0):
        sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        K = np.exp(-sq_dists / (2 * sigma ** 2))
        return K

    height, width, bands = image.shape
    X_b = image.reshape(-1, bands)
    K_b = gram_matrix_rbf(X_b, sigma=c)
    K_b_inv = np.linalg.inv(K_b)
    K_mu = np.mean(K_b, axis=0) - np.mean(K_b)

    rx_score = np.zeros((height, width))

    for i in tqdm(range(height)):
        for j in range(width):
            kernel_r = (np.linalg.norm(X_b - image[i, j, :], axis=1) ** 2) / c
            K_r = kernel_r - np.mean(kernel_r)
            diff = K_r - K_mu
            rx_score[i, j] = diff @ K_b_inv @ diff

    return rx_score
