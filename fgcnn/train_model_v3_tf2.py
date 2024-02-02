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



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print('GPU is disabled')
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.debugging import experimental as tde
import random

import subprocess
import numpy as np
import argparse
import pickle
import datetime
import time

from joblib import Parallel, delayed
import multiprocessing

import model_v3_tf2 as model
import config
from model_v3_tf2 import conv_params
import json

PRINT = False
TARGETS_DIVISION = True

import metrics
import utils




SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
# tf.random.set_seed(SEED)

conf = config.load_config()
FLAGS = None
log_file = None

#Real = np.dtype(np.float64)
Real = np.dtype(np.float32)


QA = True
if QA:
    task = 'QA'
else:
    task = 'refinement'
eps = 1.0e-7

if QA:
    files = ['general', 'nodes', 'edges', 'fsm', 'bm', 'ssm', 'edges_types', 'scores']
else:
    files = ['general', 'nodes', 'edges', 'fsm', 'bm', 'ssm', 'edges_types',  'real_dirs']
CASPS = ["CASP8", "CASP9", "CASP10", "CASP11"]
CASPS_test = ["CASP12"]


class SaveLossesAndMetrics(tf.keras.callbacks.Callback):
    
    def __init__(self, name_test):
        self.name_test = name_test
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.local_pc = []
        self.local_sc = []
        self.local_kc = []
        self.local_r2s = []
        self.local_mse = []
        self.local_zs_mse = []
        self.global_pc = []
        self.global_sc = []
        self.global_kc = []
        self.global_r2s = []
        self.global_mse = []
        self.global_zs = []
        self.global_zs_mse = []

        self.val_local_pc = []
        self.val_local_sc = []
        self.val_local_kc = []
        self.val_local_r2s = []
        self.val_local_mse = []
        self.val_local_zs_mse = []
        self.val_global_pc = []
        self.val_global_sc = []
        self.val_global_kc = []
        self.val_global_r2s = []
        self.val_global_mse = []
        self.val_global_zs = []
        self.val_global_zs_mse = []

        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.local_pc.append(logs.get('output_1_pearson_correlation'))
        self.local_sc.append(logs.get('output_1_spearman_correlation'))
        self.local_kc.append(logs.get('output_1_kendall_tau_correlation'))
        self.local_r2s.append(logs.get('output_1_r2_score'))
        self.local_mse.append(logs.get('output_1_mse_score'))
        self.local_zs_mse.append(logs.get('output_1_zscore_mse'))
        self.global_pc.append(logs.get('output_2_pearson_correlation_global'))
        self.global_sc.append(logs.get('output_2_spearman_correlation_global'))
        self.global_kc.append(logs.get('output_2_kendall_tau_correlation_global'))
        self.global_r2s.append(logs.get('output_2_r2_score_global'))
        self.global_mse.append(logs.get('output_2_mse_score_global'))
        self.global_zs.append(logs.get('output_2_z_score_global'))
        self.global_zs_mse.append(logs.get('output_2_zscore_mse_global'))

        self.val_local_pc.append(logs.get('val_output_1_pearson_correlation'))
        self.val_local_sc.append(logs.get('val_output_1_spearman_correlation'))
        self.val_local_kc.append(logs.get('val_output_1_kendall_tau_correlation'))
        self.val_local_r2s.append(logs.get('val_output_1_r2_score'))
        self.val_local_mse.append(logs.get('val_output_1_mse_score'))
        self.val_local_zs_mse.append(logs.get('val_output_1_zscore_mse'))
        self.val_global_pc.append(logs.get('val_output_2_pearson_correlation_global'))
        self.val_global_sc.append(logs.get('val_output_2_spearman_correlation_global'))
        self.val_global_kc.append(logs.get('val_output_2_kendall_tau_correlation_global'))
        self.val_global_r2s.append(logs.get('val_output_2_r2_score_global'))
        self.val_global_mse.append(logs.get('val_output_2_mse_score_global'))
        self.val_global_zs.append(logs.get('val_output_2_z_score_global'))
        self.val_global_zs_mse.append(logs.get('val_output_2_zscore_mse_global'))
        self.i += 1
        
        with  open(conf.LS_TRAINING_FILE + 'losses_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.losses)

        with  open(conf.LS_TRAINING_FILE + 'losses_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_losses)
                
        with  open(conf.LS_TRAINING_FILE + 'pc_local_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_pc)
        with  open(conf.LS_TRAINING_FILE + 'sc_local_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_sc)
        with  open(conf.LS_TRAINING_FILE + 'kc_local_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_kc)
        with  open(conf.LS_TRAINING_FILE + 'r2c_local_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_r2s)
        with  open(conf.LS_TRAINING_FILE + 'mse_local_train_'+self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_mse)
        with  open(conf.LS_TRAINING_FILE + 'mse_zs_local_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.local_zs_mse)
        with  open(conf.LS_TRAINING_FILE + 'pc_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_pc)
        with  open(conf.LS_TRAINING_FILE + 'sc_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_sc)
        with  open(conf.LS_TRAINING_FILE + 'kc_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_kc)
        with  open(conf.LS_TRAINING_FILE + 'r2c_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_r2s)
        with  open(conf.LS_TRAINING_FILE + 'mse_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_mse)
        with  open(conf.LS_TRAINING_FILE + 'mse_zs_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_zs_mse)
        with  open(conf.LS_TRAINING_FILE + 'zs_global_train_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.global_zs)   


        with  open(conf.LS_TRAINING_FILE + 'pc_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_pc)
        with  open(conf.LS_TRAINING_FILE + 'sc_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_sc)
        with  open(conf.LS_TRAINING_FILE + 'kc_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_kc)
        with  open(conf.LS_TRAINING_FILE + 'r2c_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_r2s)
        with  open(conf.LS_TRAINING_FILE + 'mse_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_mse)
        with  open(conf.LS_TRAINING_FILE + 'mse_zs_local_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_local_zs_mse)
        with  open(conf.LS_TRAINING_FILE + 'pc_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_pc)
        with  open(conf.LS_TRAINING_FILE + 'sc_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_sc)
        with  open(conf.LS_TRAINING_FILE + 'kc_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_kc)
        with  open(conf.LS_TRAINING_FILE + 'r2c_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_r2s)
        with  open(conf.LS_TRAINING_FILE + 'mse_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_mse)
        with  open(conf.LS_TRAINING_FILE + 'mse_zs_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_zs_mse)
        with  open(conf.LS_TRAINING_FILE + 'zs_global_val_'+ self.name_test + '.npy', 'wb') as f__:
            np.save(f__,self.val_global_zs)







def get_input_shapes(num_degrees, num_radii, num_features, resgap = 5, use_aggregation_tensors = True):
    shapes = []
    shapes += [tf.TensorShape((None, l//2+1, num_radii, num_features))  for l in range(2*num_degrees)]
    if use_aggregation_tensors:
        shapes += [tf.TensorShape((None, None))]*2
        shapes += [tf.TensorShape((None, 401 + resgap))]
        shapes += [tf.TensorShape((None, l//4+1, l//4+1)) for l in range(4*num_degrees)]
        shapes += [tf.TensorShape((None, num_radii, (l//2%(num_degrees))+1, (l//2//(num_degrees))+1)) for l in range(2*num_degrees**2)]
        shapes += [tf.TensorShape((None, l//4+1, l//4+1)) for l in range(4*num_degrees)]
    shapes += [tf.TensorShape((None, 3))]
    shapes += [tf.TensorShape((None, 1))]
    shapes += [tf.TensorShape((3))]
    return shapes

def get_input_shapes_lists(num_degrees, num_radii, num_features, resgap = 5, use_aggregation_tensors = True):
    shapes = []
    shapes += [[None, l//2+1, num_radii, num_features]  for l in range(2*num_degrees)]
    if use_aggregation_tensors:
        shapes += [[None, None]]*2
        shapes += [[None, 401 + resgap]]
        shapes += [[None, l//4+1, l//4+1] for l in range(4*num_degrees)]
        shapes += [[None, num_radii, (l//2%(num_degrees))+1, (l//2//(num_degrees))+1] for l in range(2*num_degrees**2)]
        shapes += [[None, l//4+1, l//4+1] for l in range(4*num_degrees)]
    shapes += [[None, 3]]
    shapes += [[None, 1]]
    shapes += [[3]]
    return shapes





def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean, std = tf.nn.moments(var, axes = [0,1])
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', std)
       
    
                
                    
                





def train(restore = False, conv = None, test = "test", num_eret = 20, num_et = 20, num_degrees = 10, num_radii = 10, num_features = 167,  num_retypes = 10,learning_rate = 0.00001, max_step =10000, batch_size = 1, radius = 10, radius2 = 12, maxQ = 1, mha = False, alpha = 1, gamma = 1, scoretype = "cad", beta = 0.2, only_count_variables = False, resgap = 2, targets_train_val_split = True, add_sph_nodes = False, multiplication = False, find_mean_std = False, non_linear_edge_retyper = False, non_linear_atom_retyper = False, nlar_second_order = False, start_epoch = 1, normalization_bool = False, sigma = 1, use_edge_retyping_in_aggregation = True,  use_diagonal_filter_if_possible = False):
    maxQ = num_radii*np.pi/radius
    


    save_path = conf.SAVE_DIR + test + '_min/model.ckpt'
    add_solvent = (num_features==168)

    init_time = time.time() 
    num_cores = multiprocessing.cpu_count()
    input_size = 4
    apply_gradients = batch_size//input_size
    print(apply_gradients)
    train_structs, val_structs = utils.make_list_train_val_strs() 
    train_structs_order = np.random.permutation(list(train_structs.keys()))
    val_structs_order = np.random.permutation(list(val_structs.keys()))
    
    num_str = train_structs_order.size
    
    EPOCHS = 100
    steps_per_epoch = num_str//(input_size*EPOCHS)


    use_multiprocessing = True
    use_aggregation_tensors = False
    for layer_i  in conv.layers:
        if layer_i.find('ConvAgg') > -1:
            use_aggregation_tensors = True
    print(use_aggregation_tensors)
    max_queue_size = 10
    with tf.Graph().as_default():
        
        
        if PRINT:
            print('Model needs to be build...', flush=True)
        input_shapes = get_input_shapes(num_degrees, num_radii, num_features, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors)
        input_shapes_lists = get_input_shapes_lists(num_degrees, num_radii, num_features, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors)
        def get_tftype(a):
            if a:
                return tf.int64
            return tf.float32
        input_array = [tf.keras.Input(shape=ish, dtype = get_tftype((it_sh+1)%len(input_shapes)==0 or (it_sh+3)%len(input_shapes)==0)) for it_sh, ish in enumerate(input_shapes*input_size)]
        input_types = [get_tftype((it_sh+1)%len(input_shapes)==0 or (it_sh+3)%len(input_shapes)==0) for it_sh, ish in enumerate(input_shapes*input_size)]
        METRICS = [metrics.CustomMetric(name = "pearson_correlation" , py_fun = metrics.pearson_correlation_multidim), metrics.CustomMetric(name = "spearman_correlation", py_fun = metrics.spearman_correlation_multidim), metrics.CustomMetric(name = "kendall_tau_correlation", py_fun = metrics.kendall_tau_correlation_multidim), metrics.CustomMetric(name = "r2_score", py_fun = metrics.r2_score_multidim),  metrics.CustomMetric(name = "mse_score", py_fun = metrics.mse_multidim),  metrics.CustomMetric(name = "zscore_mse", py_fun = metrics.zscore_mse_multidim)]
        METRICS = {'output_1':METRICS, 'output_2': [metrics.CustomMetric(name = "pearson_correlation_global" , py_fun = metrics.pearson_correlation_multidim), metrics.CustomMetric(name = "spearman_correlation_global", py_fun = metrics.spearman_correlation_multidim), metrics.CustomMetric(name = "kendall_tau_correlation_global", py_fun = metrics.kendall_tau_correlation_multidim), metrics.CustomMetric(name = "r2_score_global", py_fun = metrics.r2_score_multidim),  metrics.CustomMetric(name = "mse_score_global", py_fun = metrics.mse_multidim), metrics.CustomMetric(name = "z_score_global", py_fun = metrics.z_score_multidim),  metrics.CustomMetric(name = "zscore_mse_global", py_fun = metrics.zscore_mse_multidim)]}
        graph_nn = model.FunctionalGraphNetwork(name = 'FunctionalGraphNetwork', 
                                                num_degrees = num_degrees, 
                                                num_radii = num_radii,
                                                num_eret=num_eret,
                                                retype_dims = num_retypes,
                                                num_et = 401+resgap,
                                                model_params = conv, 
                                                ns_sample = len(input_shapes) - 2,
                                                task = 'QA' ,
                                                mha = mha,
                                                beta=beta,
                                                apply_gradients =  apply_gradients,
                                                use_aggregation_tensors = use_aggregation_tensors,
                                                add_sph_nodes = add_sph_nodes, 
                                                find_mean_std = find_mean_std, 
                                                non_linear_edge_retyper = non_linear_edge_retyper, 
                                                non_linear_atom_retyper = non_linear_atom_retyper,
                                                nlar_second_order=nlar_second_order,
                                                normalization_bool = normalization_bool,
                                                use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation,
                                                use_diagonal_filter_if_possible = use_diagonal_filter_if_possible)
            


            
        _ = graph_nn(input_array)
        graph_nn.summary(print_fn=lambda x: print("The number of variables: ", x.split(" ")[-1]) if x.split(" ")[0] == 'Total' else x)
        graph_nn.summary()
        if only_count_variables:
            return 0    
            
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,name='Adam')
        def custom_loss_function(y_true, y_pred):
            squared_difference1 = tf.square(y_true - y_pred)
            term11 = tf.reduce_mean(squared_difference1)
            y_true_zscore = (y_true - tf.math.reduce_mean(y_true))/tf.math.reduce_std(y_true)
            y_pred_zscore = (y_pred - tf.math.reduce_mean(y_pred))/tf.math.reduce_std(y_pred)
            squared_difference2 = tf.square(y_true_zscore - y_pred_zscore)
            term12 = tf.math.reduce_std(y_true)**2*tf.reduce_mean(squared_difference2) 
            term1 = gamma*term11 + (1-gamma)*term12
            term2 = tf.reduce_sum(y_true*squared_difference1)/tf.reduce_sum(y_true)
            return alpha*term1+(1-alpha)*term2
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=conf.SAVE_DIR + test+ "/model.ckpt",
            save_weights_only=True,
            monitor='val_loss',#'val_output_2_z_score_global',
            mode='min',
            save_best_only=False,
            save_freq = 'epoch' )
        model_checkpoint_callback_min = tf.keras.callbacks.ModelCheckpoint(
            filepath=conf.SAVE_DIR + test+ "_min/model.ckpt",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_freq = 'epoch' )

        save_callback = SaveLossesAndMetrics(test)
        graph_nn.compile(optimizer=opt, loss={'output_1':custom_loss_function, 'output_2':custom_loss_function}, loss_weights = {'output_1': 0.999, 'output_2': 0.001}, metrics=METRICS)
            
        rm_tb_command = ['rm ' + os.path.join(conf.TENSORBOARD_PATH,'train', '*') ]
        subprocess.call(rm_tb_command, shell=True)
        rm_tb_command = ['rm ' + os.path.join(conf.TENSORBOARD_PATH,'validate', '*') ]
        subprocess.call(rm_tb_command, shell=True)
           
            


        if restore:
            if os.path.exists(conf.SAVE_DIR + test + '_min') and len(os.listdir(conf.SAVE_DIR + test + '_min')) > 0 :
            #if os.path.exists(conf.SAVE_DIR + test + '_fit'):
                print('Restore existing model: %s' % (save_path))
                graph_nn.load_weights(save_path)# + 'model.ckpt')
            else:
                print("There is no saved model ",  save_path) 
        
        
        def get_train_sample_iter(i_b, model_file ):
            output_str = ""
            input_train = None
            output_str += "Creating the data sample N"+str(i_b) + '\n'
            time0 = time.time()
            str_file, folder = utils.create_dataset(model_file = model_file, num_degrees = num_degrees, name = test, add_solvent=add_solvent, add_sph_nodes = add_sph_nodes, usebesmatrices = not multiplication, sample_num = i_b, radius = radius, radius2 = radius2,  maxQ=maxQ, sigma = sigma, scoretype = scoretype, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors)
            time1 = time.time()
            data_files = [folder + f for f in files]
            time0 = time.time()
            input_train, [N, E] = utils.open_files(data_files, num_degrees, num_radii, num_features, sample_num = i_b, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors, add_sph_nodes = add_sph_nodes, multiplication = multiplication, usebesmatrices = not multiplication)
            time1 = time.time()
            for df in data_files:
                del_command = ["rm", df + str(i_b)]
                subprocess.call(del_command, stdout = open(folder  + "del_out" + str(i_b), "w"), stderr= open(folder  + "del_err" + str(i_b), "w"))
            return input_train
       

        
        
        
        def train_iterator():
            ti = 0
            num_s = 0
            for model_file in np.random.permutation(train_structs_order):
                ti +=1
                if ti == max_queue_size:
                    ti = 0
                if num_s%input_size == 0:
                    X = []
                    Y = np.zeros((0,1))
                    Y_mean = np.zeros((0,1))
                input_data =  get_train_sample_iter(ti, model_file )
                
                if input_data is None:
                    continue
                else:
                    num_s += 1
                    X += input_data[0]
                    Y = np.append(Y, input_data[1], axis = 0)
                    Y_mean = np.append(Y_mean, np.reshape(np.mean(input_data[1], axis = 0),[1,1]), axis =  0)
                    #input_data[1] = {'output_1':input_data[1], 'output_2':np.reshape(np.mean(input_data[1], axis = 0),[1,1])}
                    Y_dict = {'output_1':Y, 'output_2':Y_mean}
                    if num_s%input_size == 0:
                        output = tuple([tuple(X), Y_dict])
                        yield output
        
        
        
        
        def get_valid_sample_iter(i_b, model_file):
            str_file_val, folder_val = utils.create_dataset(model_file = model_file, num_degrees = num_degrees, name = test, add_solvent=add_solvent, add_sph_nodes = add_sph_nodes, usebesmatrices = not multiplication, sample_num = i_b, radius = radius, radius2 = radius2, maxQ=maxQ, sigma = sigma,  scoretype=scoretype, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors)

            data_files = [folder_val + f for f in files]
            input_val, [N, E] = utils.open_files(data_files, num_degrees, num_radii, num_features, sample_num = i_b, resgap = resgap, use_aggregation_tensors = use_aggregation_tensors, add_sph_nodes = add_sph_nodes, multiplication = multiplication, usebesmatrices = not multiplication)
            for df in data_files:
                del_command = ["rm", df + str(i_b)]
                subprocess.call(del_command, stdout = open(folder_val  + "del_out" + str(i_b), "w"), stderr= open(folder_val  + "del_err" + str(i_b), "w"))
            return input_val
        
        
        def valid_iterator():
            vi = 0
            num_s = 0
            #val_structs_order = np.random.permutation(val_structs_order)
            for model_file in np.random.permutation(val_structs_order):
                vi +=1
                if vi == max_queue_size:
                    vi = 0
                if num_s%input_size == 0:
                    X = []
                    Y = np.zeros((0,1))
                    Y_mean = np.zeros((0,1))
                input_data =  get_valid_sample_iter(vi, model_file )
                
                if input_data is None:
                    continue
                else:
                    num_s += 1
                    X += input_data[0]
                    Y = np.append(Y, input_data[1], axis = 0)
                    Y_mean = np.append(Y_mean, np.reshape(np.mean(input_data[1], axis = 0),[1,1]), axis = 0)
                    Y_dict = {'output_1':Y, 'output_2':Y_mean}
                    if num_s%input_size == 0:
                        output = tuple([tuple(X), Y_dict])
                        yield output

        
        output_shapes_dict = {}
        for i, is_i in enumerate(input_shapes):
            output_shapes_dict['input_' + str(i+1)] = is_i
        output_shapes_dict['output_1'] = tf.TensorShape((None, 1))
        output_shapes_dict['output_2'] = tf.TensorShape((None, 1))
        output_types_dict = {}
        for i, is_i in enumerate(input_types):
            output_types_dict['input_' + str(i+1)] = is_i
        output_types_dict['output_1'] = tf.float32
        output_types_dict['output_2'] = tf.float32

        #output_shapes_set = (set(input_shapes), {'output_1':tf.TensorShape((None, 1)), 'output_2':tf.TensorShape((None, 1))})
        x_shapes = tuple([tuple(is_i) for is_i in input_shapes_lists*input_size])
        #print(x_shapes)
        output_shapes_set = (x_shapes, {'output_1':[None, 1], 'output_2':[None, 1]})
        output_types_set = (tuple(input_types), {'output_1':tf.float32, 'output_2':tf.float32})
        train_dataset = tf.data.Dataset.from_generator(train_iterator, output_types = output_types_set, output_shapes = output_shapes_set).batch(1)
        valid_dataset = tf.data.Dataset.from_generator(valid_iterator, output_types = output_types_set, output_shapes = output_shapes_set).batch(1)

        _ = graph_nn.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset, verbose=1, use_multiprocessing=use_multiprocessing, callbacks=[model_checkpoint_callback, model_checkpoint_callback_min, save_callback], max_queue_size = max_queue_size, workers = num_cores, steps_per_epoch = steps_per_epoch, validation_steps = steps_per_epoch, initial_epoch=start_epoch)

       



def main():
    if FLAGS.conv != '':
        with open(FLAGS.conv, 'rb') as input:
            conv_file = pickle.load(input)
            
    else:
        if not FLAGS.decrease_dimty_in_fp:
            output_channels = []
            for i_l in range(FLAGS.num_layers_sp):
                output_channels.append(int(FLAGS.num_retypes*(5/FLAGS.num_retypes)**((i_l+1)/(FLAGS.num_layers_sp))))
            layers = []
            for i_l in range(FLAGS.num_layers_fp):
                if FLAGS.multiplication:
                    layers.append('ConvMult_'+str(FLAGS.num_retypes))
                else:
                    if i_l == 0:
                        layers.append('Conv_'+str(FLAGS.num_retypes))
                    else:
                        layers.append('ConvAgg_'+str(FLAGS.num_retypes))
        else:
            output_channels = []
            for i_l in range(FLAGS.num_layers_sp):
                output_channels.append(int((FLAGS.num_retypes//2)*(5/(FLAGS.num_retypes//2))**((i_l+1)/(FLAGS.num_layers_sp))))
            layers = []
            for i_l in range(FLAGS.num_layers_fp):
                if FLAGS.multiplication:
                    layers.append('ConvMult_'+str(int(FLAGS.num_retypes*(1/2)**((i_l+1)/FLAGS.num_layers_fp))))
                else:
                    if i_l == 0:
                        layers.append('Conv_'+str(int(FLAGS.num_retypes*(1/2)**((i_l+1)/FLAGS.num_layers_fp))))
                    else:
                        layers.append('ConvAgg_'+str(int(FLAGS.num_retypes*(1/2)**((i_l+1)/FLAGS.num_layers_fp))))
        conv_file = model.conv_params(layers, output_channels)
    print(conv_file.layers)

    train(restore = FLAGS.restore, conv = conv_file, test = FLAGS.test, num_eret = FLAGS.num_groups, num_et = 403, num_degrees = FLAGS.num_degrees, num_radii = FLAGS.num_radii, num_features = FLAGS.num_features, batch_size = FLAGS.batch_size ,num_retypes = FLAGS.num_retypes,learning_rate = FLAGS.learning_rate, max_step =FLAGS.steps, radius = FLAGS.sphere_radius, radius2 = FLAGS.neighboring_dist, mha = FLAGS.mha, alpha = FLAGS.alpha, gamma = FLAGS.gamma, scoretype = FLAGS.score_type, beta = FLAGS.beta, only_count_variables = FLAGS.count_vars, resgap = FLAGS.resgap, targets_train_val_split = FLAGS.targets_train_val_split, add_sph_nodes = FLAGS.add_sph_nodes, multiplication = FLAGS.multiplication, find_mean_std = FLAGS.find_mean_std, non_linear_edge_retyper = FLAGS.non_linear_edge_retyper, non_linear_atom_retyper = FLAGS.non_linear_atom_retyper, nlar_second_order = FLAGS.nlar_second_order, start_epoch = FLAGS.start_epoch, normalization_bool = FLAGS.normalization, sigma = FLAGS.sigma, use_edge_retyping_in_aggregation = FLAGS.use_edge_retyping_in_aggregation,  use_diagonal_filter_if_possible = FLAGS.use_diagonal_filter_if_possible)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        default=102000,
        help='Number of steps of training'
    )
    
    
    parser.add_argument(
        '-r',
        '--learning_rate',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '-j',
        '--sphere_radius',
        type=float,
        default=10.0,
        help='sphere radius'
    )
    parser.add_argument(
        '-n',
        '--neighboring_dist',
        type=float,
        default=12.0,
        help='neighboring distance'
    )
    parser.add_argument(
        '-t',
        '--test',
        type=str,
        default='',
        help='id of the test'
    )

    parser.add_argument(
        '-c',
        '--conv',
        type=str,
        default='',
        help='conv'
    )
    
    parser.add_argument(
        '--restore',
        dest='restore',
        action='store_true',
        help='if restore'
    )
    parser.add_argument(
        '--no-restore',
        dest='restore',
        action='store_false',
        help='if no restore'
    )
    parser.set_defaults(restore=True)
    parser.add_argument(
        '-l',
        '--num_degrees',
        type=int,
        default=10,
        help='Number of degrees'
    )
    parser.add_argument(
        '-p',
        '--num_radii',
        type=int,
        default=10,
        help='Number of radii'
    )
    parser.add_argument(
        '-i',
        '--num_features',
        type=int,
        default=167,
        help='Number of features'
    )

    parser.add_argument(
        '-g',
        '--num_groups',
        type=int,
        default=3,
        help='Number of groupes'
    )

    parser.add_argument(
        '-d',
        '--num_retypes',
        type=int,
        default=15,
        help='Number of retypes'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=1,
        help='Batch size'
    )
    parser.add_argument(
        '-a',
        '--mha',
        dest='mha',
        action='store_true',
        help='Multi Head Attention, bool'
    )
    parser.set_defaults(mha=False)
    parser.add_argument(
        '-y',
        '--alpha',
        type=float,
        default=1,
        help='Coefficient for the loss function'
    )
    parser.add_argument(
        '-q',
        '--score_type',
        type=str,
        default='cad',
        help='Score type'
    )
    parser.add_argument(
        '-u',
        '--beta',
        type=float,
        default=0.2,
        help='Beta for dropout'
    )
    parser.add_argument(
        '-x',
        '--count_vars',
        dest='count_vars',
        action='store_true',
        help='If only count variables, bool'
    )
    parser.set_defaults(count_vars=False)
    parser.add_argument(
        '-z',
        '--resgap',
        type=int,
        default=5,
        help='Residues gap'
    )
    parser.add_argument(
        '-f',
        '--maxQ',
        type=float,
        default=1,
        help='Max rho in Fourier space'
    )
    parser.add_argument(
        '-e',
        '--targets_train_val_split',
        dest='targets_train_val_split',
        default=0,
        help='Train/Validation split overs targets'
    )
    parser.add_argument(
        '-m',
        '--gamma',
        type=float,
        default=1,
        help='Coefficient for the loss function 2'
    )
    
    parser.add_argument(
        '--find_mean_std',
        dest='find_mean_std',
        action='store_true',
        help='Use NN to find mean and std of scores for a structure'
    )
    parser.set_defaults(find_mean_std = False)

    parser.add_argument(
        '--non_linear_atom_retyper',
        dest='non_linear_atom_retyper',
        default='store_true',
        help='Non linear atom retyper'
    )
    parser.set_defaults(non_linear_atom_retyper = False)

    parser.add_argument(
        '--nlar_second_order',
        dest='nlar_second_order',
        action='store_true',
        help='NLAR second order'
    )
    parser.set_defaults(nlar_second_order = False)

    parser.add_argument(
        '--non_linear_edge_retyper',
        dest='non_linear_edge_retyper',
        action = 'store_true',
        help='Non linear edge retyper'
    )
    parser.set_defaults(non_linear_edge_retyper = False)

    parser.add_argument(
        '--start_epoch',
        type=int,
        default=0,
        help='Start epoch'
    )

    parser.add_argument(
        '--no_normalization',
        dest='normalization',
        action='store_false',
        help='If the convolution block doesnt normalize signals'
    )
    parser.set_defaults(normalization=True)
    parser.add_argument(
        '--sigma',
        type=float,
        default=1,
        help='Blur width'
    )

    parser.add_argument(
        '--num_layers_fp',
        type=int,
        default=0,
        help='Num layers in the first part'
    )

    parser.add_argument(
        '--num_layers_sp',
        type=int,
        default=3,
        help='Num layers in the second part'
    )
    parser.add_argument(
        '--decrease_dimty_in_fp',
        dest='decrease_dimty_in_fp',
        action='store_true',
        help='Decrease dimensionality in the first part'
    )
    parser.set_defaults(decrease_dimty_in_fp=False)
    parser.add_argument(
        '--no_use_edge_retyping_in_aggregation',
        dest='use_edge_retyping_in_aggregation',
        action='store_false',
        help='If aggregation should use edge retyping'
    )
    parser.set_defaults(use_edge_retyping_in_aggregation=True)

    parser.add_argument(
        '--use_diagonal_filter_if_possible',
        dest='use_diagonal_filter_if_possible',
        action='store_true',
        help='If convolution should use diagonal filters if out_filters==in_filters'
    )
    parser.set_defaults(use_diagonal_filter_if_possible=False)
    

    parser.add_argument(
        '--multiplication',
        dest='multiplication',
        action='store_true',
        help='Use multiplication instead of aggregation on the exchange step'
    )
    parser.set_defaults(multiplication=False)

    parser.add_argument(
        '--add_sph_nodes',
        dest='add_sph_nodes',
        action='store_true',
        help='Use spherical nodes between nodes'
    )
    parser.set_defaults(add_sph_nodes=False)

    

    
    FLAGS = parser.parse_args()
    main()









    
