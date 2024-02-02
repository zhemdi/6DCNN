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


import time
import subprocess
import numpy as np
import config
import os

conf = config.load_config()
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


Real = np.dtype(np.float32)
PRINT = False


def make_list_train_val_strs():
    train_structs = {}
    val_structs = {}
    for casp in CASPS:
        casp_dir = conf.STRUCTURES_FILE_PATH + casp + '/MODELS/'
        targets_list = sorted(os.listdir(casp_dir))
        for i_t, target in enumerate(targets_list):
            #if i_t%10 != 0:
            #    print(target)
            target_dir = casp_dir + target + '/'
            target_str = conf.STRUCTURES_FILE_PATH + casp + '/TARGETS/' + target + '.pdb'
            if not os.path.exists(target_str):
                continue 
            target_dir_list = [t for t in sorted(os.listdir(target_dir)) if t[-4:] != '.sco' and t[-5:] != '.lddt' and os.path.exists(target_dir + t + '.lddt') ]
            # print(target_dir_list)
            for model in target_dir_list:
                model_file = target_dir + model
                if i_t%10 == 0:
                    val_structs[model_file] = 0
                else: 
                    train_structs[model_file] = 0
    return train_structs, val_structs



def create_dataset(num_degrees = 5, name = None, add_solvent = False, add_sph_nodes = False, usebesmatrices = False, sample_num = 0, radius = 10.0,  radius2 = 12.0, maxQ = 1,  sigma = 1, scoretype = 'cad', resgap = 2, model_file = None, use_aggregation_tensors = True):
    if model_file is not None:
        fcd = conf.DATA_FILE_PATH
        fcd += name + "/"
        target_file = conf.STRUCTURES_FILE_PATH + model_file.split('/')[-4] + '/TARGETS/' + model_file.split('/')[-2] + '.pdb'
        if not os.path.exists(fcd):
            cr_command = ["mkdir", fcd]
            subprocess.call(cr_command)
        for f in files:
            del_command = ["rm", fcd  + f + str(sample_num)]
            subprocess.call(del_command, stdout = open(fcd  + "del_out", "w"), stderr= open(fcd  + "del_err", "w"))
        create_dataset_command = [conf.MAP_GENERATOR_PATH, "--mode", "sh", "-i", model_file, "-t", target_file, "-g", fcd + "general" +str(sample_num), "-x", fcd + "nodes"+str(sample_num), "-b", fcd + "bm"+str(sample_num), "-f", fcd + "fsm"+str(sample_num), "-s", fcd + "ssm"+str(sample_num), "-e", fcd + "edges"+str(sample_num), "-y", fcd + "edges_types"+str(sample_num), "-d",  fcd + "real_dirs"+str(sample_num), "--sph_nodes", fcd + "sph_nodes"+str(sample_num), "-c", fcd + "scores"+str(sample_num), "-p", str(num_degrees), "-r", str(radius), "-n", str(radius2), "--sigma", str(sigma), "-j", scoretype, "-z", str(resgap), "-q", str(maxQ) ] 
        if add_solvent:
            create_dataset_command.append('--addsolv')
            # print(" ".join(create_dataset_command))
        if add_sph_nodes:
            create_dataset_command.append('--addshnodes')
        if not use_aggregation_tensors:
            create_dataset_command.append('--useaggtensor')
        if not usebesmatrices:
            create_dataset_command.append('--usebesmatrices')
        create_dataset_command.append("--skip_errors")
        subprocess.call(create_dataset_command, stdout = open(fcd + "create_ds_out", "w"), stderr = open(fcd + "create_ds_err", "w"))
        return model_file, fcd 
    
    


# from scipy.sparse import bsr_matrix


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



def open_files(data_files, num_degrees, num_radii, num_features, sample_num = 0, resgap = 2, use_aggregation_tensors = True, add_sph_nodes = False, multiplication = False, usebesmatrices = False):
    init_time = time.time()
    L = num_degrees - 1
    P = num_radii
    I = num_features
    if PRINT:
        print("Parsing...")
    try:
        time0 = time.time()
        with open(data_files[0]+ str(sample_num), 'rb') as general_bytestream:
            
            N = _read64(general_bytestream)
            E = _read64(general_bytestream)
        time1 = time.time()
        if PRINT:
            print("Parsing the general file: ", time1 - time0) 
    except:
        if PRINT:
            print("The general file has not been parsed", flush = True)
        return None, [None,None]
    
      
    
    
    
    try:
        with open(data_files[1] + str(sample_num), 'rb') as nodes_bytestream:
            dt = Real
            dtsize = dt.itemsize
            nodes = [np.zeros((N, l//2+1, P, I), dtype=dt) for l in range(2*(L+1))]
            time0 = time.time()
            nodes_elements = np.frombuffer(nodes_bytestream.read(dtsize*(L+1)*(L+2)*N*I*P), dtype=dt)
            time1 = time.time()
            if PRINT:
                print("Reading the nodes file: ", time1 - time0)
            if np.isnan(np.mean(nodes_elements)):
                return None, [None, None]
            nodes_elements = np.reshape(nodes_elements, [N, I, P, (L+1)*(L+2)])
            for l in range(L+1):
                nodes[2*l  ]      = np.einsum('nipl->nlpi', nodes_elements[:,:,:,l*(l+1):(l+1)*(l+1)])
                nodes[2*l+1][:,1:] = np.einsum('nipl->nlpi', nodes_elements[:,:,:,(l+1)*(l+1)+1:(l+2)*(l+1)])
            time2 = time.time()
            if PRINT:
                print("Parsing the nodes file: ", time2 - time1)
    except Exception as e:
        if PRINT:
            print(e)
            print("The nodes file has not been parsed", flush = True)
        return None, [None, None]
    

    
    try:
        with open(data_files[2] + str(sample_num), 'rb') as edges_bytestream:
            dt = np.dtype(np.uint64)
            dtsize = dt.itemsize
            edges = np.zeros((E, 2), dtype = np.uint64)
            time0 = time.time()
            edges_elements =  np.frombuffer(edges_bytestream.read(dtsize*(2*E)), dtype=dt)
            time1 = time.time()
            if PRINT:
                print("Reading the edge file: ", time1 - time0)
            edges = np.reshape(edges_elements, [E,2])
            time2 = time.time()
            if PRINT:
                print("Parsing the edge file: ", time2 - time1)
    except Exception as e:
        if PRINT:
            print(e)
            print("The edge file has not been parsed", flush = True)
        return None, [None, None]



    
    
    if add_sph_nodes or multiplication:
        try:
            with open(data_files[7] + str(sample_num), 'rb') as sph_nodes_bytestream:
                dt = Real
                dtsize = dt.itemsize
                sph_nodes = [np.zeros((E,l//2), dtype=dt) for l in range(2*(L+1))]
                bess_funs = np.zeros((E,L+1, P), dtype=dt) 
                time0 = time.time()
                sphnodes_elements = np.frombuffer(sph_nodes_bytestream.read(dtsize*E*((L+1)*(L+2) + P*(L+1))), dtype=dt)
                time1 = time.time()
                if PRINT:
                    print("Reading the sph nodes file: ", time1 - time0)
                if np.isnan(np.mean(sphnodes_elements)):
                    return None, [None, None]
                sphnodes_elements = np.reshape(sphnodes_elements, [E, ((L+1)*(L+2) + P*(L+1)) ])
                temp = np.zeros((E,0))
                for l in range(L+1):
                    sphnodes_l = np.reshape(sphnodes_elements[:, l*(l+1):(l+1)*(l+2)], [E, l+1])
                    sph_nodes[2*l  ][:, :]        = sphnodes_l[:,0::2]
                    sph_nodes[2*l+1][:,1:]        = sphnodes_l[:,1::2]
                    temp = np.concatenate(temp, sph_nodes[2*l  ], sph_nodes[2*l+1], axis = 1)
                bess_funs = np.reshape(sphnodes_elements[:, (L+1)*(L+2):], [E, P, L+1])
                time2 = time.time()
                if PRINT:
                    print("Parsing the sph nodes file: ", time2 - time1)
        except Exception as e:
            if PRINT:
                print(e)
                print("Sph nodes file has not been parsed", flush = True)
            return None, [None, None]
        try:
            indices = []
            norm0 = np.zeros((N, (L+1)**2, 401 + resgap), dtype = np.float32)
            norm1 = np.zeros((N, (L+1)**2, 401 + resgap), dtype = np.float32)

            with open(data_files[6] + str(sample_num), 'rb') as edges_types_bytestream:
                dt = np.dtype(np.int32)
                dtsize = dt.itemsize
                time0 = time.time()
                edges_types =  np.frombuffer(edges_types_bytestream.read(dtsize*(E)), dtype=dt)
                time1 = time.time()
                if PRINT:
                    print("Reading the edges types file: ", time1 - time0)
                et_matrix = np.zeros((E,401 + resgap), dtype=dt)
                for i_e in range(E):
                    linktype = edges_types[i_e]%401
                    numres = edges_types[i_e]//401
                    # print(linktype, numres, edges_types[i_e])
                    et_matrix[i_e, linktype] = 1
                    et_matrix[i_e,400+numres] = 1
                    for l in range((L+1)**2):
                        indices.append([edges[i_e,0], edges[i_e,1], l, linktype])
                    norm0[edges[i_e,0], :, linktype] += temp**2
                    norm1[edges[i_e,1], :, linktype] += temp**2
                    for l in range((L+1)**2):
                        indices.append([edges[i_e,0], edges[i_e,1], l, 400+numres]) 
                    norm0[edges[i_e,0], :, 400+numres] += temp**2
                    norm1[edges[i_e,1], :, 400+numres] += temp**2
                time2 = time.time()
                if PRINT:
                    print("Parsing the edges types file: ", time2 - time1)
        except Exception as e:
            if PRINT:
                print(e)
                print("The edges types file has not been parsed", flush = True)
            return None, [None, None]
    
        time0 = time.time()
        indices = np.reshape(np.array(indices, dtype = np.int64), [2*E, 4])
        values = np.reshape(np.ones((2*E, 1), dtype = np.float32) * np.reshape(temp.dtype(np.float32), [1,(L+1)**2]), [-1,1])
        values[:,0] /= norm0[indices[:, 0], indices[ :, 2], indices[ :, 3]]**0.5
        values[:,0] /= norm1[indices[:, 1], indices[ :, 2], indices[ :, 3]]**0.5
        time1 = time.time()
        if PRINT:
            print("Building the adjacency matrices:", time1 -time0)
        

    else:
        try:
            indices = []
            norm0 = np.zeros((N, 401 + resgap), dtype = np.float32)
            norm1 = np.zeros((N, 401 + resgap), dtype = np.float32)
            with open(data_files[6] + str(sample_num), 'rb') as edges_types_bytestream:
                dt = np.dtype(np.int32)
                dtsize = dt.itemsize
                time0 = time.time()
                edges_types =  np.frombuffer(edges_types_bytestream.read(dtsize*(E)), dtype=dt)
                time1 = time.time()
                if PRINT:
                    print("Reading the edges types file: ", time1 - time0)
                et_matrix = np.zeros((E,401 + resgap), dtype=dt)
                for i_e in range(E):
                    linktype = edges_types[i_e]%401
                    numres = edges_types[i_e]//401
                    # print(linktype, numres, edges_types[i_e])
                    et_matrix[i_e, linktype] = 1
                    et_matrix[i_e,400+numres] = 1
                    indices.append([edges[i_e,0], edges[i_e,1], linktype])
                    norm0[edges[i_e,0], linktype] += 1
                    norm1[edges[i_e,1], linktype] += 1
                    indices.append([edges[i_e,0], edges[i_e,1], 400+numres]) 
                    norm0[edges[i_e,0], 400+numres] += 1
                    norm1[edges[i_e,1], 400+numres] += 1
                time2 = time.time()
                if PRINT:
                    print("Parsing the edges types file: ", time2 - time1)
        except Exception as e:
            if PRINT:
                print(e)
                print("The edges types file has not been parsed", flush = True)
            return None, [None, None]
    
        time0 = time.time()
        indices = np.reshape(np.array(indices, dtype = np.int64), [2*E, 3])
        values = np.ones((2*E, 1), dtype = np.float32)
        values[:,0] /= norm0[indices[:, 0], indices[ :, 2]]**0.5
        values[:,0] /= norm1[indices[:, 1], indices[ :, 2]]**0.5
        time1 = time.time()
        if PRINT:
            print("Building the adjacency matrices:", time1 -time0)




    if use_aggregation_tensors:
        time0 = time.time()
        
        edges1 = edges

        edges = []
        edges_senders = np.zeros( ( E, N), dtype=dt)
        edges_receivers = np.zeros(( N, E), dtype=dt)
            
        edges_receivers[ edges1[:,1], np.arange(edges1[:,1].size)] = 1.0
        edges_senders[np.arange(edges1[:,0].size), edges1[:,0]] = 1.0
        
        edges.append(edges_senders)
        edges.append(edges_receivers)
        time1 = time.time()
        if PRINT:
            print("Building the edges matrices:", time1 -time0)
    
    
        try:
            with open(data_files[3] + str(sample_num), 'rb') as first_slater_bytestream:
                dt = Real
                dtsize = dt.itemsize
                fsm = [np.zeros((E, l//4+1, l//4+1), dtype=dt) for l in range(4*(L+1))]
                time0 = time.time()
                fsm_elements = np.frombuffer(first_slater_bytestream.read(dtsize*E*int(round((L+1)*(L+2)*(4/3*L+1)))), dtype=dt)
                time1 = time.time()
                if PRINT:
                    print("Reading the first rotation file: ", time1 - time0)
                if np.isnan(np.mean(fsm_elements)):
                    return None, [None, None]
                fsm_elements = np.reshape(fsm_elements, [E, int(round((L+1)*(L+2)*(4/3*L+1))) ])
                for l in range(L+1):
                    fsm_l = np.reshape(fsm_elements[:, int(round((l*(l+1)*(4/3*l-1/3)))): int(round((l*(l+1)*(4/3*l-1/3))))+2*(l+1)*(2*l+1)], [E, l+1, 2*(2*l+1)])
                    fsm[4*l  ]        = fsm_l[:,:,0::4]
                    fsm[4*l+1]        = fsm_l[:,:,1::4]
                    fsm[4*l+2][:,:,1:] = fsm_l[:,:,2::4]
                    fsm[4*l+3][:,:,1:] = fsm_l[:,:,3::4]
                time2 = time.time()
                if PRINT:
                    print("Parsing the first rotation file: ", time2 - time1)
        except Exception as e:
            if PRINT:
                print(e)
                print("First rotation file has not been parsed", flush = True)
            return None, [None, None]
        if usebesmatrices:
            try:
                with open(data_files[4] + str(sample_num), 'rb') as bessel_bytestream:
                    dt = Real
                    dtsize = dt.itemsize
                    bm = [np.zeros(( E, P, L+1 - l//2, L+1 - l//2), dtype=dt) for l in range(2*(L+1))]
                    time0 = time.time()
                    bm_elements = np.frombuffer(bessel_bytestream.read(dtsize*E*P*int(round((L+1)*(L+2)*(2/3*L+1)))), dtype=dt)
                    time1 = time.time()
                    if PRINT:
                        print("Reading the transition file: ", time1 - time0)
                    if np.isnan(np.mean(bm_elements)):
                        return None, [None, None]
                    bm_elements = np.reshape(bm_elements, [E, P, int(round((L+1)*(L+2)*(2/3*L+1)))])
                    for l in range(L+1):
                        bm_l = np.reshape(bm_elements[:,:,int(round((l)/3*(6*L**2+2*l**2+13-9*l+18*L-6*L*l))): int(round((l+1)/3*(6*L**2+2*l**2+6-5*l+12*L-6*L*l))) ], [E,P,(L+1-l),2*(L+1-l)])
                        bm[2*l  ] = bm_l[:,:,:, ::2]
                        bm[2*l+1] = bm_l[:,:,:,1::2]
                    time2 = time.time()
                    if PRINT:
                        print("Parsing the transition file: ", time2 - time1)
            except Exception as e:
                if PRINT:
                    print(e)
                    print("Transition file has not been parsed", flush = True)
                #print(E, P, L, E*P*int((L+1)*(L+2)*(2/3*L+1)), itr)
                return None, [None, None]
            time0 = time.time()
            bm1 = bm
            bm = [np.zeros((E, P, (l//2)%(L+1)+1, (l//2)//(L+1)+1), dtype=dt) for l in range(2*(L+1)**2)]
            for m in range(L+1):
                for l1 in range(m, L+1):
                    for l2 in range(m, L+1):
                        bm[2*(l1*(L+1)+l2)  ][:,:,m,m] =  bm1[2*m  ][:,:,l2-m,l1-m]
                        bm[2*(l1*(L+1)+l2)+1][:,:,m,m] =  bm1[2*m+1][:,:,l2-m,l1-m]
            del bm1
            time1 = time.time()
            if PRINT:
                print("Transforming Bessel matrices: ", time1 - time0)

            try:
                with open(data_files[5] + str(sample_num), 'rb') as second_slater_bytestream:
                    dt = Real
                    dtsize = dt.itemsize
                    ssm = [np.zeros(( E, l//4+1, l//4+1), dtype=dt) for l in range(4*(L+1))]
                    time0 = time.time()
                    ssm_elements = np.frombuffer(second_slater_bytestream.read(dtsize*E*int(round((L+1)*(L+2)*(4/3*L+1)))), dtype=dt)
                    time1 = time.time()
                    if PRINT:
                        print("Reading the second rotation file: ", time1 - time0)
                    if np.isnan(np.mean(ssm_elements)):
                        return None, [None, None]
                    ssm_elements = np.reshape(ssm_elements, [E, int(round((L+1)*(L+2)*(4/3*L+1))) ])
                    for l in range(L+1):
                        ssm_l = np.reshape(ssm_elements[:, int(round(l*(l+1)*(4/3*l-1/3))): int(round(l*(l+1)*(4/3*l-1/3)))+2*(l+1)*(2*l+1)], [E, l+1, 2*(2*l+1)])
                        ssm[4*l  ]        = ssm_l[:,:,0::4]
                        ssm[4*l+1]        = ssm_l[:,:,1::4]
                        ssm[4*l+2][:,:,1:] = ssm_l[:,:,2::4]
                        ssm[4*l+3][:,:,1:] = ssm_l[:,:,3::4]
                    time2 = time.time()
                    if PRINT:
                        print("Parsing the second rotation file: ", time2 - time1)
            except Exception as e:
                if PRINT:
                    print(e)
                    print("Second rotation file has not been parsed", flush = True)
                return None, [None, None]


    if QA:
        try:
            with open(data_files[-1] + str(sample_num), 'rb') as scores_bytestream:
                dt = np.dtype(np.int32)
                dtsize = dt.itemsize
                real_scores = np.zeros((N,1))
                time0 = time.time()
                scores_elements = np.frombuffer(scores_bytestream.read(dtsize*N), dtype=dt).astype(float)
                time1 = time.time()
                if PRINT:
                    print("Reading the scores file: ", time1 - time0)
                if np.isnan(np.mean(scores_elements)):
                    return None, [None, None]
                real_scores[:,0] = scores_elements/1000000
                time2 = time.time()
                if PRINT:
                    print("Parsing the scores file: ", time2 - time1)
        except Exception as e:
            if PRINT:
                print(e)
                print("Scores file has not been parsed", flush = True)
            return None, [None, None]
        
        
        
    else:
        try:
            with open(data_files[-1] + str(sample_num), 'rb') as directions_bytestream:
                dt = Real
                dtsize = dt.itemsize
                real_dirs = np.zeros((N,3))
                dirs_elements = np.frombuffer(directions_bytestream.read(dtsize*3*N), dtype=dt)
                if np.isnan(np.mean(dirs_elements)):
                    return None, [None, None]
                real_dirs = np.reshape(dirs_elements,[N,3])
            real_dirs /= np.max(np.linalg.norm(real_dirs, axis = 1))
        except Exception as e:
            if PRINT:
                print(e)
                print("Directions file has not been parsed", flush = True)
            return None, [None, None]

    final_time = time.time()
    if PRINT:
        print("All files are parsed: ", final_time - init_time)        
    
    
    X = [np.float32(n) for n in nodes]
    if use_aggregation_tensors:
        X += [np.float32(e) for e in edges] + [np.float32(et_matrix)] + [np.float32(f) for f in fsm]
        if usebesmatrices:
            X += [np.float32(b) for b in bm] + [np.float32(s) for s in ssm]
    
    if multiplication:
        X += [np.float32(bess_funs)]
    if add_sph_nodes or multiplication:
        X += [np.float32(sphn_i) for sphn_i in sph_nodes]
    
    X += [np.int64(indices)] +  [np.float32(values)] 
    if add_sph_nodes:
        X += [np.array([N,N,(L+1)**2,401+resgap], dtype = np.int64)]
    else:
        X += [np.array([N,N,401+resgap], dtype = np.int64)]
    
    if QA:
        Y = np.float32(real_scores)
    else:
        Y = np.float32(real_dirs)
    return [X,Y], [N, E]