'''
Copyright (C) 2021  Dmitrii Zhemchuzhnikov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.debugging import experimental as tde
import numpy as np


def pearson_correlation(arr1, arr2):
    pearson = np.corrcoef(arr1, arr2).astype(np.float32)[0,1]
    return pearson

def pearson_correlation_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    pc_arr = np.array([pearson_correlation(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(pc_arr, arr_shape[:-2])




def rankdata(arr):
    ranks = np.zeros(arr.size)
    ranks[np.argsort(arr)] = np.arange(len(arr)) + 1
    return ranks


def spearman_correlation(arr1, arr2):
    ranks1 = rankdata(arr1)
    ranks2 = rankdata(arr2)
    spearman = np.corrcoef(ranks1, ranks2).astype(np.float32)[0,1]
    return spearman

def spearman_correlation_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    sc_arr = np.array([spearman_correlation(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(sc_arr, arr_shape[:-2])



def kendall_tau_correlation(arr1, arr2):
    ranks1 = rankdata(arr1)
    ranks2 = rankdata(arr2)
    concordant = 0
    discordant = 0
    N = arr1.size
    for i in range(N):
        for j in range(i):
            t = (arr1[i] - arr1[j])*(arr2[i] - arr2[j])
            if t > 0:
                concordant += 1
            elif t < 0:
                discordant += 1
    num_pairs = N*(N+1)/2
    kendall_tau = (concordant - discordant)/num_pairs
    return kendall_tau

def kendall_tau_correlation_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    kc_arr = np.array([kendall_tau_correlation(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(kc_arr, arr_shape[:-2])





def r2_score(arr1, arr2):
    sstot = np.sum((arr1 - np.mean(arr1))**2)
    ssres = np.sum((arr1 - arr2)**2)
    r2 = 1 - ssres/sstot
    return r2

def r2_score_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    r2c_arr = np.array([r2_score(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(r2c_arr, arr_shape[:-2]) 


def mse_metrics(arr1, arr2):
    mse_value = np.mean((arr1-arr2)**2)
    return mse_value




def mse_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    mse_arr = np.array([mse_metrics(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(mse_arr, arr_shape[:-2])


def z_score_mse(arr1, arr2):
    zscore1 = (arr1 - np.mean(arr1))/np.std(arr1)
    zscore2 = (arr2 - np.mean(arr2))/np.std(arr2)
    return np.std(arr1)**2*mse_metrics(zscore1, zscore2)


def zscore_mse_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    zscore_mse_arr = np.array([z_score_mse(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(zscore_mse_arr, arr_shape[:-2])

def z_score(arr1, arr2):
    temp_arr = arr1[(arr1 - np.mean(arr1))/np.std(arr1) > -2]
    return ((arr1 - np.mean(temp_arr))/np.std(temp_arr))[np.argmax(arr2)]

def z_score_multidim(arr1, arr2):
    arr_shape = arr1.shape
    last_dim = arr_shape[-2]
    resh_arr1 = np.reshape(arr1, [-1, last_dim])
    resh_arr2 = np.reshape(arr2, [-1, last_dim])
    zscore_arr = np.array([z_score(r1, r2) for r1, r2 in zip(resh_arr1, resh_arr2)]).astype(np.float32)
    return np.reshape(zscore_arr, arr_shape[:-2])


class CustomMetric(tf.keras.metrics.Metric):

    def __init__(self, py_fun = pearson_correlation,  name="custom_metric", **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)

        self.py_fun = py_fun
        self.custom_metric = self.add_weight( name=name, initializer="zeros")
        self.cm_list = []
        self.i = 0

    def update_state(self, y_true, y_pred, sample_weight=None):     
        current_value = tf.numpy_function(self.py_fun, [y_true, y_pred], tf.float32)
        self.cm_list.append(current_value)
        self.custom_metric.assign(tf.reduce_mean(tf.stack(self.cm_list)))
        self.i += 1

    def result(self):
        return self.custom_metric