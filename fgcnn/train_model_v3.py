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


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python import debug as tf_debug
from tensorflow.debugging import experimental as tde
from tensorflow.python.client import timeline
import random
import os
import subprocess
import numpy as np
import argparse
import pickle


import model_v3 as model
import config
from model import conv_params

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)
# tf.random.set_seed(SEED)

conf = config.load_config()
FLAGS = None
log_file = None

QA = True
create_dataset = True
# create_dataset = False
eps = 1.0e-7

# Real = np.dtype(np.float32)
Real = np.dtype(np.float64)

if QA:
    files = ['general', 'nodes', 'edges', 'fsm', 'bm', 'ssm',  'scores']
else:
    files = ['general', 'nodes', 'edges', 'fsm', 'bm', 'ssm',  'real_dirs']
# CASPS = ["CASP7", "CASP8", "CASP9", "CASP10", "CASP11", "CASP12"]
# CASPS = ["CASP11"]
CASPS = ["CASP8", "CASP9", "CASP10", "CASP11"]
CASPS_test = ["CASP12"]
# CASPS = ["harmonics_net_dataset/small_dataset_structures"]
# CASPS = ["small_dataset"]

# CASPS = ["very_small_dataset"]



def _read32(bytestream):
  dt = np.dtype(np.uint32)
  output = np.frombuffer(bytestream.read(4), dtype=dt)
#   print(output)
  return output[0]


def _read64(bytestream):
  dt = np.dtype(np.uint64)
  output = np.frombuffer(bytestream.read(8), dtype=dt)
#   print(output)
  return int(output[0])

def choose_folder():
    
    folder = None
    while folder == None:
        caspNumber = random.choice(CASPS)
        casp_dir = conf.DATA_FILE_PATH + caspNumber + '/'
        target_dir = casp_dir + random.choice(os.listdir(casp_dir)) + '/'
        model_dir = target_dir + random.choice(os.listdir(target_dir)) + '/'
        print(model_dir)
        if_good_dir = True
        for f in files:
            if not os.path.exists(model_dir + f):
                if_good_dir = False
        if if_good_dir:
            folder = model_dir
    return folder

def choose_file_and_create_dataset(prev_str_file = " ", val = False, caspNumber = None, target_name = None, model_name = None, num_degrees = 5):
    folder = None
    
    while folder == None:
        # if caspNumber == None:
        if True:
            if not val:
                caspNumber = random.choice(CASPS)
            else:
                caspNumber = random.choice(CASPS_test)
        casp_dir = conf.STRUCTURES_FILE_PATH + caspNumber + '/MODELS/'
        # if target_name == None:
        if True:
            target_name = random.choice(os.listdir(casp_dir))
        target_dir = casp_dir + target_name + '/'
        target_file = conf.STRUCTURES_FILE_PATH + caspNumber + '/TARGETS/' + target_name + '.pdb'
        if not os.path.exists(target_file):
            print("no target file:", target_file)
            continue

        # if model_name == None:
        if True:
            model_name = random.choice(os.listdir(target_dir))
        model_file = target_dir + model_name
        if model_file[-4:] == '.sco':
            model_file = model_file[:-4]
        print(model_file, target_file)
        fcd = conf.DATA_FILE_PATH 
        if os.path.exists(model_file + '.sco'):
            chosen_file = model_file
            if chosen_file != prev_str_file:
                for f in files:
                    del_command = ["rm", conf.DATA_FILE_PATH  + f]
                    subprocess.call(del_command, stdout = open(conf.DATA_FILE_PATH  + "del_out", "w"), stderr= open(conf.DATA_FILE_PATH  + "del_err", "w"))
                # create_dataset_command = ['/home/zhemd/oriented3dCNN/Harmonics Graph Net/mapsGenerator/build/maps_generator', "-i", model_file, "-t", target_file, "-g", fcd + "general" , "-x", fcd + "nodes", "-b", fcd + "bm", "-f", fcd + "fsm", "-s", fcd + "ssm", "-e", fcd + "edges", "-d",  fcd + "real_dirs", "-c", fcd + "scores", "-p", str(FLAGS.num_degrees) ]
                create_dataset_command = ['/home/zhemd/oriented3dCNN/Harmonics Graph Net/mapsGenerator/build/maps_generator', "--mode", "sh", "-i", model_file, "-t", target_file, "-g", fcd + "general" , "-x", fcd + "nodes", "-b", fcd + "bm", "-f", fcd + "fsm", "-s", fcd + "ssm", "-e", fcd + "edges", "-d",  fcd + "real_dirs", "-c", fcd + "scores", "-p", str(num_degrees) ]
                subprocess.call(create_dataset_command, stdout = open(fcd + "create_ds_out", "w"), stderr = open(fcd + "create_ds_err", "w"))
            folder = fcd
    return chosen_file, folder
        






def placeholder_inputs(num_degrees, num_radii, num_features, num_groups):
    """
    Creates placeholder for the input data during the training of the model
    
    """
    #sparse_
    
    nodes_placeholders = [tf.placeholder(tf.float32, shape=(None, l//2+1, num_radii, num_features) ,name='nodes_degree_'+str(l)) for l in range(2*num_degrees)]
    edges_placeholders = [tf.placeholder(tf.float32, shape=(None, None) ,name='edges_'+str(g)) for g in range(2*num_groups)]
    first_slater_matrices_placeholders = [tf.placeholder(tf.float32, shape=(None, l//4+1, l//4+1) ,name='fsm_degree_'+str(l)) for l in range(4*num_degrees)]
    bessel_matrices_placeholders = [tf.placeholder(tf.float32, shape=(None, num_radii, (l//2%(num_degrees))+1, (l//2//(num_degrees))+1) ,name='bm_order_'+str(l)) for l in range(2*num_degrees**2)]
    second_slater_matrices_placeholders = [tf.placeholder(tf.float32, shape=(None, l//4+1, l//4+1) ,name='ssm_degree_'+str(l)) for l in range(4*num_degrees)]
    adjacency_matrices_placeholders = [tf.placeholder(tf.float32, shape=(None, None) ,name='adjacency_matrix_'+str(g)) for g in range(num_groups)]
    if QA:
        real_scores_placeholders = tf.placeholder(tf.float32, shape=(None, 1) ,name='real_scores')
    else:
        real_directions_placeholder = tf.placeholder(tf.float32, shape=(None, 3) ,name='real_directions')

    if QA:
        return [nodes_placeholders, edges_placeholders, first_slater_matrices_placeholders, bessel_matrices_placeholders, second_slater_matrices_placeholders, adjacency_matrices_placeholders, real_scores_placeholders]
    return [nodes_placeholders, edges_placeholders, first_slater_matrices_placeholders, bessel_matrices_placeholders, second_slater_matrices_placeholders, adjacency_matrices_placeholders, real_directions_placeholder]




def open_files(data_files, num_degrees, num_radii, num_features, num_groups, agg_norm = 'group'):
    #general_filename, nodes_filename, edges_filename, fsm_filename, bm_filename, ssm_filename, am_filename = files
    L = num_degrees - 1
    P = num_radii
    I = num_features
    G = num_groups
    print("Parsing...")
    try:
        with open(data_files[0], 'rb') as general_bytestream:
            
            N = _read64(general_bytestream)
            E = [_read64(general_bytestream) for g in range(G+1)]
    except:
        print("The general file has not been parsed")
        return None, [None,None]
    
    # print("The general file has been parsed")
       
    
    
    
    try:
        with open(data_files[1], 'rb') as nodes_bytestream:
            # dt = np.dtype(np.float64)
            # dt = np.dtype(np.float32)
            dt = Real
            dtsize = dt.itemsize
            nodes = [np.zeros((N, l//2+1, P, I)) for l in range(2*(L+1))]
            nodes_elements = np.frombuffer(nodes_bytestream.read(dtsize*(L+1)*(L+2)*N*I*P), dtype=dt)
            if np.isnan(np.mean(nodes_elements)):
                return None, [None, None]
            itr = 0
            for n in range(N):
                for i in range(I):
                    for p in range(P):
                        for l in range(L+1):
                        
                        
                            nodes_node_degree_order_radii = nodes_elements[itr:itr+2*(l+1)]
                            itr += 2*(l+1)
                            nodes[2*l][n, :, p, i] = nodes_node_degree_order_radii[::2]
                            nodes[2*l+1][n, 1:, p, i] = nodes_node_degree_order_radii[3::2]
    except:
        print("The nodes file has not been parsed")
        return None, [None, None]
   
    
    # print([np.max(np.abs(n)) for n in nodes])
    # print("The nodes file has been parsed")
    try:
        with open(data_files[2], 'rb') as edges_bytestream:
            dt = np.dtype(np.uint64)
            dtsize = dt.itemsize
            edges = [np.zeros((E[g], g//G+1), dtype = np.uint64) for g in range(G+1)]
            edges_elements =  np.frombuffer(edges_bytestream.read(dtsize*(np.sum(E)+E[-1])), dtype=dt)
            itr = 0
            for g in range(G+1):
                for e in range(E[g]):
                    # edges_group_edge = np.frombuffer(edges_bytestream.read(8*(g//G+1)), dtype=dt)
                    edges_group_edge = edges_elements[itr:itr+g//G+1]
                    itr += g//G+1
                    edges[g][e,:] = edges_group_edge
    except:
        print("The edge file has not been parsed")
        return None, [None, None]

    # print("The edge file has been parsed")
            
    adjacency_matrices = []
    for g in range(G):
        adjacency_matrix = np.zeros((N, N))
        for e in range(E[g]):
            edge_i = edges[g][e]
            sender = edges[-1][edge_i,0]
            receiver = edges[-1][edge_i,1]
            adjacency_matrix[sender, receiver] = 1
        # adjacency_matrix += np.eye(N)
        adjacency_matrix = adjacency_matrix/((np.sum(adjacency_matrix, axis =1, keepdims = True))**0.5*(np.sum(adjacency_matrix, axis =0, keepdims = True))**0.5+eps)
        adjacency_matrix += np.eye(N)
        adjacency_matrices.append(adjacency_matrix)
    
    # print("Adjacency matrices have been build")
    
    
    edges1 = edges

    edges = []
    for g in range(G):
        edges_group_senders = np.zeros( (E[-1], N))
        edges_group_receivers = np.zeros((N, E[-1]))
        
        edges_group = edges1[g]
        for e in edges_group:
            receiver = edges1[-1][e,1]
            sender = edges1[-1][e,0]
            edges_group_receivers[receiver, e] = 1.0
            edges_group_senders[e, sender] = 1.0
        #if agg_norm == 'nodes':
        #    edges_group_receivers = (edges_group_receivers)/(np.linalg.norm(edges_group_receivers, axis = 1, keepdims=True) +1)
        edges.append(edges_group_senders)
        edges.append(edges_group_receivers)
    
    # print("Edges matrices have been build")
        
    
    # print(int(E[-1]*(L+1)*(L+2)*(4/3*L+1)))
    # itr = 0
    # for e in range(E[-1]):
    #     for l in range(L+1):
    #         for m1 in range(l+1):
    #             itr += 2*(2*l+1)
    # print(itr)
    
    
    try:
        with open(data_files[3], 'rb') as first_slater_bytestream:
            # dt = np.dtype(np.float64)
            # dt = np.dtype(np.float32)
            dt = Real
            dtsize = dt.itemsize
            fsm = [np.zeros((E[-1], l//4+1, l//4+1)) for l in range(4*(L+1))]
            fsm_elements = np.frombuffer(first_slater_bytestream.read(dtsize*E[-1]*int((L+1)*(L+2)*(4/3*L+1))), dtype=dt)
            if np.isnan(np.mean(fsm_elements)):
                return None, [None, None]
            itr = 0
            for e in range(E[-1]):
                for l in range((L+1)):
                    for m1 in range(l+1):
                        # fsm_degree_order = np.frombuffer(first_slater_bytestream.read(8*(2*(l+1)+2*l)), dtype=dt)
                        fsm_degree_order = fsm_elements[itr:itr+2*(2*l+1)]
                        itr += 2*(2*l+1)
                        fsm[4*l][e, m1, :] = fsm_degree_order[::4]
                        fsm[4*l+1][e, m1, :] = fsm_degree_order[1::4]
                        fsm[4*l+2][e, m1, 1:] = fsm_degree_order[2::4]
                        fsm[4*l+3][e, m1, 1:] = fsm_degree_order[3::4]
    except:
        print("First rotation file has not been parsed")
        return None, [None, None]

    # print("First rotation file has been parsed")


    # print(E[-1]*P*(L+1)*(L+2)*(2/3*L+1))
    # itr = 0
    # for e in range(E[-1]):
    #     for p in range(P):
    #         for l in range(L+1):
    #             for m1 in range(L+1 - l):
    #                 itr += 2*(L+1 - l)
    # print(itr)


    try:
        with open(data_files[4], 'rb') as bessel_bytestream:
            # dt = np.dtype(np.float64)
            # dt = np.dtype(np.float32)
            dt = Real
            dtsize = dt.itemsize
            bm = [np.zeros((E[-1], P, L+1 - l//2, L+1 - l//2)) for l in range(2*(L+1))]
            bm_elements = np.frombuffer(bessel_bytestream.read(dtsize*E[-1]*P*int((L+1)*(L+2)*(2/3*L+1))), dtype=dt)
            if np.isnan(np.mean(bm_elements)):
                return None, [None, None]
            itr = 0
            for e in range(E[-1]):
                for p in range(P):
                    for l in range(L+1):
                        for m1 in range(L+1 - l):
                            # bm_degree_order = np.frombuffer(bessel_bytestream.read(8*2*(L+1 - l)), dtype=dt)
                            bm_degree_order = bm_elements[itr:itr+2*(L+1 - l)]
                            itr += 2*(L+1 - l)
                            bm[2*l][e, p, m1, :] = bm_degree_order[::2]
                            bm[2*l+1][e, p, m1, 0:] = bm_degree_order[1::2]
    except:
        print("Transition file has not been parsed")
        return None, [None, None]

    # print("Transition file has been parsed")


    bm1 = bm
    bm = [np.zeros((E[-1], P, (l//2)%(L+1)+1, (l//2)//(L+1)+1)) for l in range(2*(L+1)**2)]
    for m in range(L+1):
        for l1 in range(m, L+1):
            for l2 in range(m, L+1):
                bm[2*(l1*(L+1)+l2)  ][:,:,m,m] =  bm1[2*m  ][:,:,l2-m,l1-m]
                bm[2*(l1*(L+1)+l2)+1][:,:,m,m] =  bm1[2*m+1][:,:,l2-m,l1-m]

    # print("Transition matrices have been build")


    try:
        with open(data_files[5], 'rb') as second_slater_bytestream:
            # dt = np.dtype(np.float64)
            # dt = np.dtype(np.float32)
            dt = Real
            dtsize = dt.itemsize
            ssm = [np.zeros((E[-1], l//4+1, l//4+1)) for l in range(4*(L+1))]
            ssm_elements = np.frombuffer(second_slater_bytestream.read(dtsize*E[-1]*int((L+1)*(L+2)*(4/3*L+1))), dtype=dt)
            if np.isnan(np.mean(ssm_elements)):
                return None, [None, None]
            itr = 0
            for e in range(E[-1]):
                for l in range((L+1)):
                    for m1 in range(l+1):
                        # ssm_degree_order = np.frombuffer(second_slater_bytestream.read(8*(2*(l+1)+2*l)), dtype=dt)
                        ssm_degree_order = ssm_elements[itr:itr+2*(2*l+1)]
                        itr += 2*(2*l+1)
                        ssm[4*l][e, m1, :] = ssm_degree_order[::4]
                        ssm[4*l+1][e, m1, :] = ssm_degree_order[1::4]
                        ssm[4*l+2][e, m1, 1:] = ssm_degree_order[2::4]
                        ssm[4*l+3][e, m1, 1:] = ssm_degree_order[3::4]
    except:
        print("Second rotation file has not been parsed")
        return None, [None, None]


    # print("Second rotation file has been parsed")

    if QA:
        try:
            with open(data_files[6], 'rb') as scores_bytestream:
                dt = np.dtype(np.int32)
                dtsize = dt.itemsize
                real_scores = np.zeros((N,1))
                scores_elements = np.frombuffer(scores_bytestream.read(dtsize*N), dtype=dt).astype(float)
                if np.isnan(np.mean(scores_elements)):
                    return None, [None, None]
                for n in range(N):
                    score = scores_elements[n]
                    real_scores[n,:] = score/1000000
        except:
            print("Scores file has not been parsed")
            return None, [None, None]
        
        # print("Scores file has been parsed")

        
    else:
        try:
            with open(data_files[6], 'rb') as directions_bytestream:
                # dt = np.dtype(np.float64)
                # dt = np.dtype(np.float32)
                dt = Real
                dtsize = dt.itemsize
                real_dirs = np.zeros((N,3))
                dirs_elements = np.frombuffer(directions_bytestream.read(dtsize*3*N), dtype=dt)
                if np.isnan(np.mean(dirs_elements)):
                    return None, [None, None]
                for n in range(N):
                    # dir_node = np.frombuffer(directions_bytestream.read(8*3), dtype=dt)
                    dir_node = dirs_elements[3*n:3*n+2]
                    real_dirs[n,:] = dir_node

            real_dirs /= np.max(np.linalg.norm(real_dirs, axis = 1))
        except:
            print("Directions file has not been parsed")
            return None, [None, None]

        # print("Directions file has been parsed")



        
    if QA:
        return [[np.float32(n) for n in nodes], [np.float32(e) for e in edges], [np.float32(f) for f in fsm], [np.float32(b) for b in bm], [np.float32(s) for s in ssm], np.float32(adjacency_matrices), np.float32(real_scores)], [N, E]
    else:
        return [[np.float32(n) for n in nodes], [np.float32(e) for e in edges], [np.float32(f) for f in fsm], [np.float32(b) for b in bm], [np.float32(s) for s in ssm], np.float32(adjacency_matrices), np.float32(real_dirs)], [N, E]
    









def variable_summaries(var, name = 'summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean, std = tf.nn.moments(var, axes = [0,1])
        tf.summary.scalar(name + '_mean', mean)
        tf.summary.scalar(name + '_std', std)
       
    
                
                    
                





def fill_feed_dict(data_files, placeholders, num_degrees, num_radii, num_features, num_groups, agg_norm = 'group'):
    """
    :param data_set:    |
    :param maps_pl:     |
    :param meta_pl:     | Data placeholders
    :param gt_pl:       |
    :param is_training: |
    :param training: True if the model is training
    :param batch_size_: integer
    :return: dictionnary to give to sess.run(.., feed_dict = )
    """
    data, [N, E] = open_files(data_files, num_degrees, num_radii, num_features, num_groups, agg_norm = agg_norm)
    
    if data == None or N==0:
        return None
    #print(data)
    #print(placeholders)
    feed_dict = {}
    for placeholders_i, data_i  in zip(placeholders[:-1], data[:-1]):
        for p, d in zip(placeholders_i, data_i):
            feed_dict[p] = d
    feed_dict[placeholders[-1]] = data[-1]

    #feed_dict = dict(zip(placeholders, data))


    return feed_dict




def train(restore = None, conv = None, test = "test", num_degrees = 10, num_radii = 10, num_features = 167, num_groups = 3, num_retypes = 10,learning_rate = 0.00001, max_step =10000, agg_norm = 'group'):
    updates = 0
    dln = 1
    decayLoss = 0.999
    slidingloss = 0
    loss_values = []
    prev_folder = " "
    prev_str_file = " "
    str_file = " "
    mean = 0
    mean_sq = 0
    num = 0



    do_validation = True
    dln_val = 1
    slidingloss_val = 0
    loss_values_val = []
    prev_folder_val = " "
    prev_str_file_val = " "
    str_file_val = " "
    mean_val = 0
    mean_sq_val = 0
    num_val = 0

    
    with tf.Graph().as_default():
        sess = tf.Session()
        # sess = tf.compat.v1.Session()
        
        if restore is None:
            print('Model needs to be build...')
            

            graph_nn = model.GraphModel(layers_param = conv,
                num_retypes = num_retypes,
                num_degrees = num_degrees,
                num_radii = num_radii,
                num_features = num_features,
                num_groups = num_groups,
                normalization=agg_norm,
                qa=QA)

            placeholders = placeholder_inputs(num_degrees, num_radii, num_features, num_groups)

            #predictions, mn, N = graph_nn.predict(*placeholders[:-1])
            predictions, fn, num_vars = graph_nn.predict(*placeholders[:-1])

            loss = graph_nn.compute_loss(predictions, placeholders[-1])

            train_op = graph_nn.train(loss, learning_rate)

            # adding all the summaries
            

            variable_summaries(predictions, "predictions")
            variable_summaries(placeholders[-1], "ground_truth")
            with tf.name_scope('loss'):
                loss_summary = tf.summary.scalar('loss', tf.reduce_mean(loss))

            merged = tf.summary.merge_all()

            only_loss = tf.summary.merge([loss_summary])

            print(merged.name)

            rm_tb_command = ['rm ' + os.path.join(conf.TENSORBOARD_PATH,'train', '*') ]
            subprocess.call(rm_tb_command, shell=True)
            rm_tb_command = ['rm ' + os.path.join(conf.TENSORBOARD_PATH,'validate', '*') ]
            subprocess.call(rm_tb_command, shell=True)


            writer = tf.summary.FileWriter(os.path.join(conf.TENSORBOARD_PATH,'train'))
            writer.add_graph(sess.graph)

            writer_validate = tf.summary.FileWriter(os.path.join(conf.TENSORBOARD_PATH,'validate'))

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.global_variables())

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.global_variables())

            # initializing variables
            init = tf.global_variables_initializer()
            sess.run(init)
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "nanodina:6008")
            # tde.enable_dump_debug_info(
            #                         "/tmp/tensorboard/debugging",
            #                         tensor_debug_mode="FULL_HEALTH",
            #                         circular_buffer_size=-1)


        else:
            print('Restore existing model: %s' % (restore))
            print('Latest checkpoint: %s' % (tf.train.latest_checkpoint(restore)))

            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(restore) + '.meta')
            
            saver.restore(sess, tf.train.latest_checkpoint(restore))

            graph = tf.get_default_graph()

            nodes_placeholders = [graph.get_tensor_by_name('nodes_degree_'+str(l)+':0') for l in range(2*num_degrees)]
            edges_placeholders = [graph.get_tensor_by_name('edges_'+str(g)+':0') for g in range(2*num_groups)]
            first_slater_matrices_placeholders = [graph.get_tensor_by_name('fsm_degree_'+str(l)+':0') for l in range(4*num_degrees)]
            bessel_matrices_placeholders = [graph.get_tensor_by_name('bm_order_'+str(l)+':0') for l in range(2*num_degrees**2)]
            second_slater_matrices_placeholders = [graph.get_tensor_by_name('ssm_degree_'+str(l)+':0') for l in range(4*num_degrees)]
            adjacency_matrices_placeholders = [graph.get_tensor_by_name('adjacency_matrix_'+str(g)+':0') for g in range(num_groups)]
            if QA:
                real_scores_placeholders = graph.get_tensor_by_name('real_scores:0')
                placeholders = [nodes_placeholders, edges_placeholders, first_slater_matrices_placeholders, bessel_matrices_placeholders, second_slater_matrices_placeholders, adjacency_matrices_placeholders, real_scores_placeholders]
            else:
                real_directions_placeholder = graph.get_tensor_by_name('real_directions:0')
                placeholders = [nodes_placeholders, edges_placeholders, first_slater_matrices_placeholders, bessel_matrices_placeholders, second_slater_matrices_placeholders, adjacency_matrices_placeholders, real_directions_placeholder]
            predictions = graph.get_tensor_by_name("output_nodes:0")
            loss = graph.get_tensor_by_name("loss:0")
            train_op = graph.get_tensor_by_name('train_op:0')
            merged = graph.get_tensor_by_name("Merge/MergeSummary:0")


        nv = sess.run([num_vars])
        
        print("The number of variables: ", nv[0])
        feed_dict = None
        prev_feed_dict = None

        feed_dict_val = None
        prev_feed_dict_val = None
        while updates <  max_step:
            
            while feed_dict == None:
                if create_dataset:
                    print("Creating a data sample")
                    str_file, folder = choose_file_and_create_dataset(prev_str_file = prev_str_file, num_degrees = num_degrees)
                else:
                    folder = choose_folder()
                if (prev_folder != folder and not create_dataset) or (prev_str_file != str_file and create_dataset):
                
                    data_files = [folder + f for f in files]
                    print("Starting to read")
                    feed_dict = fill_feed_dict(data_files, placeholders, num_degrees, num_radii, num_features, num_groups, agg_norm = agg_norm)
                else:
                    feed_dict, prev_feed_dict = prev_feed_dict, None
            print("Feeding")
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            _, loss_value,  s = sess.run([train_op, loss,   merged], feed_dict=feed_dict)#, options=run_options, run_metadata=run_metadata)
            if np.isnan(np.mean(loss_value)):
                print("Output is NaN")
                break
            # tl = timeline.Timeline(run_metadata.step_stats)
            # print(tl.generate_chrome_trace_format(show_memory=True))
            
            len_temp = len(feed_dict[placeholders[-1]])
            mean = (mean*num + np.sum(feed_dict[placeholders[-1]]))/(num+len_temp)
            mean_sq = (mean_sq*num + np.sum(feed_dict[placeholders[-1]]**2))/(num+len_temp)
            num += len_temp
            print("Loss value: %f; Variance for the structure: %f"%(np.mean(loss_value), np.var(feed_dict[placeholders[-1]])))
            print("Current global variance: %f"%(mean_sq - mean**2))
            dln = dln * decayLoss
            updates += 1
            slidingloss = decayLoss * slidingloss + (1 - decayLoss) * np.mean(loss_value)
            print("Step %d: sliding loss  - %f" %(updates, slidingloss/ (1 - dln)) )
            writer.add_summary(s, updates)
            loss_values.append(slidingloss/ (1 - dln))
            prev_folder = folder
            prev_str_file = str_file
            prev_feed_dict, feed_dict = feed_dict, None



            if updates % 20 == 0 and do_validation:
                print("\n\n---------------------------------VALIDATION-------------------------------\n\n")
                while feed_dict_val == None:
                    if create_dataset:
                        print("Creating a data sample")
                        str_file_val, folder_val = choose_file_and_create_dataset(prev_str_file = prev_str_file_val, val = True, num_degrees = num_degrees)
                    else:
                        folder_val = choose_folder()
                    if (prev_folder_val != folder_val and not create_dataset) or (prev_str_file_val != str_file_val and create_dataset):
                    
                        data_files = [folder_val + f for f in files]
                        print("Starting to read")
                        feed_dict_val = fill_feed_dict(data_files, placeholders, num_degrees, num_radii, num_features, num_groups, agg_norm = agg_norm)
                    else:
                        feed_dict_val, prev_feed_dict_val = prev_feed_dict_val, None
                print("Feeding")
                loss_value,   s = sess.run([loss,  only_loss], feed_dict=feed_dict_val)
                if np.isnan(np.mean(loss_value)):
                    print("Output is NaN")
                    break
                len_temp = len(feed_dict_val[placeholders[-1]])
                mean_val = (mean_val*num_val + np.sum(feed_dict_val[placeholders[-1]]))/(num_val+len_temp)
                mean_sq_val = (mean_sq_val*num_val + np.sum(feed_dict_val[placeholders[-1]]**2))/(num_val+len_temp)
                num_val += len_temp
                print("Loss value: %f; Variance for the structure: %f"%(np.mean(loss_value), np.var(feed_dict_val[placeholders[-1]])))
                print("Current global variance: %f"%(mean_sq_val - mean_val**2))
                dln_val = dln_val * decayLoss
                
                slidingloss_val = decayLoss * slidingloss_val + (1 - decayLoss) * np.mean(loss_value)
                print("Step %d: sliding loss  - %f" %(updates // 20, slidingloss_val/ (1 - dln_val)) )
                writer_validate.add_summary(s, (updates))
                loss_values_val.append(slidingloss_val/ (1 - dln_val))
                prev_folder_val = folder_val
                prev_str_file_val = str_file_val
                prev_feed_dict_val, feed_dict_val = feed_dict_val, None
                print("\n\n---------------------------------TRAINING-------------------------------\n\n")



            if updates % 100 == 0:
                if not os.path.exists(conf.SAVE_DIR + test):
                    os.makedirs(conf.SAVE_DIR + test)

                save_path = saver.save(sess, conf.SAVE_DIR + test + '/model.ckpt')
                print("Model saved in path: %s" % save_path)

                f__ = open(conf.LS_TRAINING_FILE + 'losses'+ test + '.npy', 'wb')
                np.save(f__,loss_values)

                f__.close()





def main():
    
    with open(FLAGS.conv, 'rb') as input:
        conv_file = pickle.load(input)
    train(restore = FLAGS.restore, conv = conv_file, test = FLAGS.test, num_degrees = FLAGS.num_degrees, num_radii = FLAGS.num_radii, num_features = FLAGS.num_features, num_groups = FLAGS.num_groups, num_retypes = FLAGS.num_retypes,learning_rate = FLAGS.learning_rate, max_step =FLAGS.steps)#, agg_norm='nodes')
    

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
        default=None,
        type=str,
        help='path to model'
    )
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
    
    

    FLAGS = parser.parse_args()
    main()


    









    
