from visualization import create_viewer_tool
from AnomalyDetection.rx import *
import scipy.io
import matplotlib.pyplot as plt
from AnomalyDetection.DimensionalityReduction import pca, svd, mnf

img = scipy.io.loadmat('Data/San_Diego.mat')
map_data = img['map']
data = img['data']

data = GRX(pca(data, 2))

create_viewer_tool(gt=map_data, data=data)


