import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    data_size = (75, 75, 10)  # Size of the datacube (75x75 pixels, 10 spectral bands)
    anomaly_size = (5, 5, 10)  # Size of the anomaly regions (5x5 pixels, 10 spectral bands)

    background_mean = 1.0
    background_variance = 0.1
    background = np.random.normal(background_mean, np.sqrt(background_variance), data_size)

    anomaly_mean = 1.5
    anomaly_variance = 0.1
    anomaly_region_1 = np.random.normal(anomaly_mean, np.sqrt(anomaly_variance), anomaly_size)
    anomaly_region_2 = np.random.normal(anomaly_mean, np.sqrt(anomaly_variance), anomaly_size)

    background[10:15, 10:15, :] = anomaly_region_1  # First anomaly at (10,10)
    background[20:25, 20:25, :] = anomaly_region_2  # Second anomaly at (20,20)

    gt = np.zeros()

    return background


if __name__ == "__main__":
    data = generate_data()
