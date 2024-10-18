import numpy as np
from sklearn import svm
from data_utils.preprocess import normalize
from utils.metric import Metric
class SVM:
    def __init__(self, num_electrodes, num_datapoints, num_classes):
        self.svc = svm.SVC(kernel='rbf', C=1.0, gamma='scale')