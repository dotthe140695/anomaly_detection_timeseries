import numpy as np
from collections import  Counter
import scipy.io as sio
import scipy.stats
from scipy.stats import skew


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    mean = np.mean(list_values)
    absolute_mean = np.mean(np.absolute(list_values))
    std = np.std(list_values)
    skewness = skew(list_values)
    root_mean_square = np.sqrt(np.mean(np.square(list_values)))
    root = np.square(np.mean(np.sqrt(np.absolute(list_values))))
    peak_value = np.max(list_values)
    peak_to_peak = np.max(list_values) -np.min(list_values)
    shape_factor = root_mean_square/absolute_mean
    impulse_factor = np.max(list_values)/absolute_mean
    clearance_factor = np.max(list_values)/root    
    return [mean, absolute_mean, std, skewness, root_mean_square, root, 
            peak_value, peak_to_peak , shape_factor, impulse_factor, clearance_factor]


def get_statistic_features(list_values):
    entropy = calculate_entropy(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + statistics

