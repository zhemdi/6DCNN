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

import numpy as np
import tensorflow as tf


from sympy.physics.quantum.cg import CG


EPS = 1.0e-4
LEAK = 0.2



class conv_params:
    def __init__(self, layers, output_channels):
        """
        Class containing the net architecture
        """
        # Parse the layers list to check for errors
        errors = 0
        for la in layers:
            parse = la.split('_')
            if not(parse[0] == 'Conv' or parse[0] == 'ConvAgg' or parse[0] == 'ConvMult'):
                print('ERROR: Anomaly detected while parsing argument :', parse[0],'is not a valid keyword')
                errors += 1
        if not errors == 0:
            raise ValueError(str(errors) + ' error(s) while parsing the argument')
        self.layers = layers
        self.output_channels = output_channels
    def __call__(self):
        print("Layers:")
        for la in self.layers:
            print(la)
        print(self.output_channels)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)





def PrecomputingTensorsForConv6D( num_degrees = 4):
    
    L = num_degrees
    tensors = []
    coefs_N = []
    for l in range(L):
        for l1 in range(L):
            for l2 in range(np.abs(l -l1),min(L,l+l1+1)):
                coefs_N.append(8*np.pi**2/(2*l1+1)*np.sqrt(((2*l+1)*(2*l1+1))/(4*np.pi*(2*l2+1)))*float(CG(l,0,l1,0,l2,0).doit()))
                T1 = np.zeros((l+1, l1+1, l2+1))
                T2 = np.zeros((l+1, l1+1, l2+1))
                T3 = np.zeros((l+1, l1+1, l2+1))
                for k in range(0, l+1):
                    for k1 in range(0, l1+1):
                            
                            
                        if abs(k+k1) < l2+1:
                                
                            T1[k, k1, k+k1] = float(CG(l,k,l1,k1,l2,k+k1).doit())*(-1)**(k1)
                        if k1 > 0:
                               
                            if abs(k-k1) < l2+1:
                                if k -k1 >= 0:
                                   
                                    T2[k, k1, k-k1] = float(CG(l,k,l1,-k1,l2,k-k1).doit())*(-1)**(l1)
                                else:
                                   
                                    T3[k, k1, k1-k] = (-1)**(k1-k)*(-1)**(l1+l2)*float(CG(l,k,l1,-k1,l2,k-k1).doit())
                T1_tensor = tf.convert_to_tensor(T1, dtype = tf.float32)
                T2_tensor = tf.convert_to_tensor(T2, dtype = tf.float32)
                T3_tensor = tf.convert_to_tensor(T3, dtype = tf.float32)
                tensors.append(T1_tensor)
                tensors.append(T2_tensor)
                tensors.append(T3_tensor)
    
    return tensors, coefs_N





class Dropout_coefficients(tf.keras.layers.Layer):
    def __init__(self, beta = 0.1, name = 'Droupout'):
        super(Dropout_coefficients, self).__init__()
        self.beta = beta
        self.name_ch = name
        self.dropout = tf.keras.layers.Dropout(self.beta)


    def get_config(self):
        config = super(Dropout_coefficients, self).get_config()
        config.update({"beta": self.beta})
        config.update({"name_ch": self.name_ch})
        return config

    
    def call(self,input, training = None):
        output = []
        for x_i in input:
            output.append(self.dropout(x_i, training = training) )
        return output



class Retyper(tf.keras.layers.Layer):
    def __init__(self, output_channels = 10, name = 'retyper', num_degrees = 4, num_radii = 5, non_linear_atom_retyper = False, activation = 'lrelu', second_layer = False):
        super(Retyper, self).__init__()
        self.output_channels = output_channels
        self.name_ch = name
        self.L = num_degrees
        self.P = num_radii
        self.activation = activation
        self.non_linear_atom_retyper = non_linear_atom_retyper
        self.second_layer = second_layer
    
    def get_output_shapes(self):
        return [tf.TensorShape((None, l//2, self.P,  self.output_channels)) for l in range(2*self.L)]
    def compute_output_shape(self, input_shape):
        return [(None, l//2, self.P,  self.output_channels) for l in range(2*self.L)]
    
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.retyper_weights = self.add_weight("weights_" + self.name_ch , shape = [self.input_channels, self.output_channels], initializer = 'truncated_normal')
        if self.non_linear_atom_retyper:
            self.retyper_bias = self.add_weight("bias_" + self.name_ch , shape = [self.input_channels, self.output_channels], initializer = 'truncated_normal')
            if self.second_layer:
                self.retyper_weights2 = self.add_weight("weights_" + self.name_ch + "_2" , shape = [self.output_channels, self.output_channels], initializer = 'truncated_normal')
        self.built = True
        
    def get_config(self):
        config = super(Retyper, self).get_config()
        config.update({"output_channels": self.output_channels})
        config.update({"name_ch": self.name_ch})
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"activation": self.activation})
        config.update({"non_linear_atom_retyper": self.non_linear_atom_retyper})

        return config
    
    
    def call(self,input):
        output = []
        if self.non_linear_atom_retyper: 
            sq_tensor = 0.0
            for l in range(self.L):
                sq_tensor += tf.reduce_sum(tf.reduce_mean(tf.square(input[2*l]), axis = 2), axis = 1)
                sq_tensor += tf.reduce_sum(tf.reduce_mean(tf.square(input[2*l+1]), axis = 2), axis = 1)
            sq_tensor /= (self.L)**2
            mu_tensor = tf.reduce_mean(sq_tensor, axis  = 0, keepdims = True)
            tensor_shifted = sq_tensor - mu_tensor
            cov_tensor = tf.reduce_mean(tf.einsum('ne,nt->net', tensor_shifted, tensor_shifted ), axis = 0)
            self.retyper_matrix = tf.einsum('et,tr->er',  cov_tensor, self.retyper_weights ) + self.retyper_bias
            if self.activation == 'lrelu':
                self.retyper_matrix = lrelu(self.retyper_matrix, leak = LEAK)
            elif self.activation == 'tanh':
                self.retyper_matrix = tf.tanh(self.retyper_matrix)
            if self.second_layer:
                self.retyper_matrix = tf.einsum('et,tr->er', self.retyper_matrix, self.retyper_weights2 )
        else:
            self.retyper_matrix = self.retyper_weights




        for l in range(self.L):
            output.append(tf.matmul(input[2*l], self.retyper_matrix) )
            output.append(tf.matmul(input[2*l+1], self.retyper_matrix))
        return output



class NodesAggregation(tf.keras.layers.Layer):
    def __init__(self,  name = 'aggregation', num_degrees = 5, num_radii = 5, num_eret = 15, shift_frames = True, multiplication = False, use_edge_retyping_in_aggregation = True):
        super(NodesAggregation, self).__init__()
        self.name_ch = name
        self.L = num_degrees
        self.P = num_radii
        self.T = num_eret
        self.shift_frames = shift_frames
        self.multiplication = multiplication
        self.use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation

    def build(self,input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.built = True
        if self.use_edge_retyping_in_aggregation and self.shift_frames and not self.multiplication:
            self.weights_ch = self.add_weight("weights_" + self.name_ch , shape = [self.T], initializer = 'truncated_normal')

        if self.multiplication:
            self.neighborhood_function = []
            for l in range(self.L):
                if self.use_edge_retyping_in_aggregation:
                    self.neighborhood_function.append(self.add_weight("neighborhood_function_coefs_" + str(l) + "_real_" + self.name_ch , shape = [l+1,self.P, self.T], initializer = 'truncated_normal'))
                    self.neighborhood_function.append(self.add_weight("neighborhood_function_coefs_" + str(l) + "_imag_" + self.name_ch , shape = [l+1,self.P, self.T], initializer = 'truncated_normal'))
                else:
                    self.neighborhood_function.append(self.add_weight("neighborhood_function_coefs_" + str(l) + "_real_" + self.name_ch , shape = [l+1,self.P], initializer = 'truncated_normal'))
                    self.neighborhood_function.append(self.add_weight("neighborhood_function_coefs_" + str(l) + "_imag_" + self.name_ch , shape = [l+1,self.P], initializer = 'truncated_normal'))
        # super(NodesAggregation, self).build(input_shapes)
    def get_output_shapes(self):
        return [tf.TensorShape((None, self.T, l//2, self.P,  self.input_channels)) for l in range(2*self.L)] 

    def compute_output_shape(self, input_shape):
        return [(None, self.T, l//2, self.P,  self.input_channels) for l in range(2*self.L)] 

    def get_config(self):
        config = super(NodesAggregation, self).get_config()
        config.update({"name_ch": self.name_ch})
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"T": self.T})
        return config
    
    def call(self,input_array):
        t = 0
        input = input_array[t:t+2*self.L]
        t += 2*self.L
        self.groups_of_edges = input_array[t:t + 2]
        t += 2
        self.edges_types = input_array[t]
        t += 1
        self.first_slater_matrices = input_array[t:t + 4*self.L]
        t += 4*self.L
        self.bessel_matrices = input_array[t:t + 2*self.L**2]
        t += 2*self.L**2
        self.second_slater_matrices = input_array[t:t + 4*self.L]
        t += 4*self.L
        if self.multiplication:
            self.bess_funs = input_array[t]
            t += 1
            self.sph_nodes = input_array[t:t + 2*self.L]
            t += 2*self.L
        
        output = []
        N_to_E = self.groups_of_edges[0]
        E_to_N = self.groups_of_edges[1]
        first_rotation_and_translation = []
        for l1 in range(self.L):
            reshaped_real  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*self.input_channels])), [-1, l1+1, self.P*self.input_channels])
            reshaped_imag  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*self.input_channels])), [-1, l1+1, self.P*self.input_channels])
            fsm_l_pos_real = tf.transpose(self.first_slater_matrices[4*l1  ], [0,1,2])
            fsm_l_pos_imag = tf.transpose(self.first_slater_matrices[4*l1+1], [0,1,2])
            fsm_l_neg_real = tf.transpose(self.first_slater_matrices[4*l1+2], [0,1,2])
            fsm_l_neg_imag = tf.transpose(self.first_slater_matrices[4*l1+3], [0,1,2])
            rotated_real   = tf.matmul(fsm_l_pos_real, reshaped_real) - tf.matmul(fsm_l_pos_imag, reshaped_imag)
            rotated_imag   = tf.matmul(fsm_l_pos_real, reshaped_imag) + tf.matmul(fsm_l_pos_imag, reshaped_real)
            rotated_real  += (-1)**l1*(tf.matmul(fsm_l_neg_real, reshaped_real) + tf.matmul(fsm_l_neg_imag, reshaped_imag))
            rotated_imag  += (-1)**l1*(-tf.matmul(fsm_l_neg_real, reshaped_imag) + tf.matmul(fsm_l_neg_imag, reshaped_real))

            if not self.shift_frames or self.multiplication:
                first_rotation_and_translation.append(tf.reshape(rotated_real, [-1, l1+1, self.P* self.input_channels]))
                first_rotation_and_translation.append(tf.reshape(rotated_imag, [-1, l1+1, self.P* self.input_channels]))
            else:
                reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(rotated_real, [-1, l1+1, self.P, self.input_channels]), [0,2,1,3]), [-1, l1+1, self.input_channels])
                reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(rotated_imag, [-1, l1+1, self.P, self.input_channels]), [0,2,1,3]), [-1, l1+1, self.input_channels])
                    
                    
                for l2 in range(self.L):
                    bm_real = tf.reshape(self.bessel_matrices[2*(l1*self.L+l2)  ], [-1, l2+1, l1+1])
                    bm_imag = tf.reshape(self.bessel_matrices[2*(l1*self.L+l2)+1], [-1, l2+1, l1+1])
                    term_dd_real = tf.matmul(bm_real, reshaped_transposed_real) + tf.matmul(bm_imag, reshaped_transposed_imag)
                    term_dd_imag = tf.matmul(bm_real, reshaped_transposed_imag) - tf.matmul(bm_imag, reshaped_transposed_real)

                    term_dd_real = tf.reshape(tf.transpose(tf.reshape(term_dd_real, [-1, self.P, l2+1, self.input_channels]), [0,2,1,3]),[-1, l2+1, self.P*self.input_channels])
                    term_dd_imag = tf.reshape(tf.transpose(tf.reshape(term_dd_imag, [-1, self.P, l2+1, self.input_channels]), [0,2,1,3]),[-1, l2+1, self.P*self.input_channels])


                    if len(first_rotation_and_translation) < 2*(l2+1):
                        first_rotation_and_translation.append(term_dd_real)
                        first_rotation_and_translation.append(term_dd_imag)
                    else:
                        first_rotation_and_translation[2*l2] += term_dd_real
                        first_rotation_and_translation[2*l2+1] += term_dd_imag
        
        if not self.shift_frames and not self.multiplication:
            if self.use_edge_retyping_in_aggregation:
                for l in range(self.L):
                    term_real = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l  ], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,et,eli->nli', E_to_N, self.edges_types, first_rotation_and_translation[2*l  ], [-1, l+1, self.P, self.input_channels]))
                    term_imag = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l+1], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,et,eli->nli', E_to_N, self.edges_types, first_rotation_and_translation[2*l+1], [-1, l+1, self.P, self.input_channels]))
                    output.append( term_real)
                    output.append( term_imag)
            else:
                for l in range(self.L):
                    term_real = tf.reshape(input[2*l  ], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, first_rotation_and_translation[2*l  ], [-1, l+1, self.P, self.input_channels]))
                    term_imag = tf.reshape(input[2*l+1], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, first_rotation_and_translation[2*l+1], [-1, l+1, self.P, self.input_channels]))
                    output.append( term_real)
                    output.append( term_imag)
            return output
        if self.multiplication:
            if self.use_edge_retyping_in_aggregation:
                edges_weights_real = []
                for l in range(self.L):
                    edges_weights_real += [(1 + 1*(l>0))*tf.reshape(tf.einsum('elpt,el->ept',self.neighborhood_function[2*l  ], self.sph_nodes[2*l  ]), [-1,1,self.P]) - tf.reshape(tf.einsum('elpt,el->ept',self.neighborhood_function[2*l+1], self.sph_nodes[2*l+1]), [-1,1,self.P])] 
                    

                edges_weights_real = tf.concat(edges_weights_real, axis = 1)
                edges_weights_real = tf.reshape(tf.einsum('elpt,elp->et', edges_weights_real, self.bess_funs), [-1,1,1])
                for l in range(self.L):
                    first_rotation_and_translation[2*l  ] = first_rotation_and_translation[2*l  ]*edges_weights_real - first_rotation_and_translation[2*l+1]*edges_weights_imag
                    term_real = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l  ], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,et,et,eli->ntli', E_to_N, self.edges_types, edges_weights_real, first_rotation_and_translation[2*l  ]), [-1, self.T, l+1, self.P, self.input_channels])
                    term_imag = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l+1], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,et,et,eli->ntli', E_to_N, self.edges_types, edges_weights_real, first_rotation_and_translation[2*l+1]), [-1, self.T, l+1, self.P, self.input_channels])
                    output.append( term_real)
                    output.append( term_imag)
            else:
                edges_weights_real = []
                for l in range(self.L):
                    edges_weights_real += [(1 + 1*(l>0))*tf.reshape(tf.einsum('elp,el->ep',self.neighborhood_function[2*l  ], self.sph_nodes[2*l  ]), [-1,1,self.P]) - tf.reshape(tf.einsum('elp,el->ep',self.neighborhood_function[2*l+1], self.sph_nodes[2*l+1]), [-1,1,self.P])] 
                edges_weights_real = tf.concat(edges_weights_real, axis = 1)
                edges_weights_real = tf.reshape(tf.einsum('elp,elp->e', edges_weights_real, self.bess_funs), [-1,1,1])
                for l in range(self.L):
                    first_rotation_and_translation[2*l  ] = first_rotation_and_translation[2*l  ]*edges_weights_real 
                    first_rotation_and_translation[2*l+1] = first_rotation_and_translation[2*l+1]*edges_weights_real
                    term_real = tf.reshape(input[2*l  ], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, first_rotation_and_translation[2*l  ]), [-1, l+1, self.P, self.input_channels])
                    term_imag = tf.reshape(input[2*l+1], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, first_rotation_and_translation[2*l+1]), [-1, l+1, self.P, self.input_channels])
                    output.append( term_real)
                    output.append( term_imag)
            return output



        for l in range(self.L):
            second_rotation_real  = tf.matmul(self.second_slater_matrices[4*l], first_rotation_and_translation[2*l]) - tf.matmul(self.second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1])
            second_rotation_imag  = tf.matmul(self.second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) + tf.matmul(self.second_slater_matrices[4*l+1], first_rotation_and_translation[2*l])
            second_rotation_real += (-1)**l*( tf.matmul(self.second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) + tf.matmul(self.second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1]))
            second_rotation_imag += (-1)**l*(-tf.matmul(self.second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) + tf.matmul(self.second_slater_matrices[4*l+3], first_rotation_and_translation[2*l]))
            if self.use_edge_retyping_in_aggregation:
                term_real = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l  ], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('et,ne,eli->ntli', self.edges_types, E_to_N, second_rotation_real), [-1, self.T, l+1, self.P, self.input_channels])
                term_imag = tf.reshape(self.weights_ch, [1, self.T, 1, 1, 1]) * tf.reshape(input[2*l+1], [-1, 1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('et,ne,eli->ntli', self.edges_types, E_to_N, second_rotation_imag), [-1, self.T, l+1, self.P, self.input_channels])
            else:
                term_real = tf.reshape(input[2*l  ], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, second_rotation_real), [-1, l+1, self.P, self.input_channels])
                term_imag = tf.reshape(input[2*l+1], [-1, l+1, self.P, self.input_channels]) + tf.reshape(tf.einsum('ne,eli->nli', E_to_N, second_rotation_imag), [-1, l+1, self.P, self.input_channels])
            
            
            output.append( term_real)
            output.append( term_imag)
        return output


class Convolution6D(tf.keras.layers.Layer):
    def __init__(self, name = 'convolution_6D', output_channels=10, num_degrees = 4, num_radii = 5, num_eret = 15, after_aggregation = True, use_edge_retyping_in_aggregation = True, use_diagonal_filter_if_possible = False):
        super(Convolution6D, self).__init__()
        self.name_ch = name
        self.L = num_degrees
        self.P = num_radii
        self.T = num_eret
        self.after_aggregation = after_aggregation
        self.tensors, self.coefs_N = PrecomputingTensorsForConv6D( self.L)
        self.output_channels = output_channels
        self.use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation
        self.use_diagonal_filter_if_possible = use_diagonal_filter_if_possible
    def build(self,input_shapes):

        self.input_channels = input_shapes[0].as_list()[-1]
        self.weights_ch_list = []
        self.biases_list = []
        
        
            

        for l in range(self.L):
            if self.after_aggregation and self.use_edge_retyping_in_aggregation:
                if self.use_diagonal_filter_if_possible and self.input_channels == self.output_channels:
                    weights_degree_real = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_real", shape = [l+1, self.T, self.P, self.output_channels], initializer = 'truncated_normal')
                    weights_degree_imag = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_imag", shape = [l+1, self.T, self.P, self.output_channels], initializer = 'truncated_normal')
                    

                    temp_term = np.ones((l+1, self.T, self.P, 1))
                    temp_term[0,:,:,:] = 0
                else:
                    weights_degree_real = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_real", shape = [l+1, self.T, self.P, self.input_channels, self.output_channels], initializer = 'truncated_normal')
                    weights_degree_imag = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_imag", shape = [l+1, self.T, self.P, self.input_channels, self.output_channels], initializer = 'truncated_normal')
                    

                    temp_term = np.ones((l+1, self.T, self.P, 1, 1))
                    temp_term[0,:,:,:,:] = 0
            else:
                if self.use_diagonal_filter_if_possible and self.input_channels == self.output_channels:
                    weights_degree_real = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_real", shape = [l+1, self.P, self.output_channels], initializer = 'truncated_normal')
                    weights_degree_imag = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_imag", shape = [l+1, self.P, self.output_channels], initializer = 'truncated_normal')
                    

                    temp_term = np.ones((l+1, self.P,  1))
                    temp_term[0,:,:] = 0
                else:
                    weights_degree_real = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_real", shape = [l+1, self.P, self.input_channels, self.output_channels], initializer = 'truncated_normal')
                    weights_degree_imag = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_imag", shape = [l+1, self.P, self.input_channels, self.output_channels], initializer = 'truncated_normal')
                    

                    temp_term = np.ones((l+1, self.P, 1, 1))
                    temp_term[0,:,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            self.weights_ch_list.append(weights_degree_real)
            self.weights_ch_list.append(weights_degree_imag)

            
            bias_degree_real =  self.add_weight("biases_" + self.name_ch + "_" + str(l) + "_real", shape = [1, l+1, self.P, self.output_channels], initializer = 'random_uniform')
            bias_degree_imag =  self.add_weight("biases_" + self.name_ch + "_" + str(l) + "_imag", shape = [1, l+1, self.P, self.output_channels], initializer = 'random_uniform')
            temp_term_bias = np.ones((1, l+1, self.P,1))
            temp_term_bias[:,0,:,:] = 0
            temp_term_bias = tf.convert_to_tensor(temp_term_bias, dtype = tf.float32)
            bias_degree_imag = bias_degree_imag*temp_term_bias
            self.biases_list.append(bias_degree_real)
            self.biases_list.append(bias_degree_imag)
        self.built = True
        
    
    def get_output_shapes(self):
        return [tf.TensorShape((None, l//2, self.P,  self.output_channels)) for l in range(2*self.L)]
    
    def compute_output_shape(self, input_shape):
        return [(None, l//2, self.P,  self.output_channels) for l in range(2*self.L)]
    
    
    def get_config(self):
        config = super(Convolution6D, self).get_config()
        config.update({"name_ch": self.name_ch})
        config.update({"L": self.L})
        config.update({"T": self.T})
        config.update({"P": self.P})
        config.update({"tensors": self.tensors})
        config.update({"coefs_N": self.coefs_N})
        config.update({"output_channels": self.output_channels})
        return config
    
    
    def call(self, input):
        output = []
        itr = 0
        for l in range(self.L):
            output_i_real = 0.0
            output_i_imag = 0.0
            for l1 in range(self.L):
                for l2 in range(np.abs(l -l1),min(self.L,l+l1+1)):
                    coef = self.coefs_N[itr]
                    if self.after_aggregation and self.use_edge_retyping_in_aggregation:
                        if self.use_diagonal_filter_if_possible and self.input_channels == self.output_channels:
                            output_i_real += coef*tf.einsum('lmk,ntkpi,mtpi->nlpi', (self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1  ] )
                            output_i_real +=-coef*tf.einsum('lmk,ntkpi,mtpi->nlpi', (self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,ntkpi,mtpi->nlpi', (self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,ntkpi,mtpi->nlpi', (self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1  ] )
                        else:
                            output_i_real += coef*tf.einsum('lmk,ntkpi,mtpij->nlpj', (self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1  ] )
                            output_i_real +=-coef*tf.einsum('lmk,ntkpi,mtpij->nlpj', (self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,ntkpi,mtpij->nlpj', (self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,ntkpi,mtpij->nlpj', (self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1  ] )
                    else:
                        if self.use_diagonal_filter_if_possible and self.input_channels == self.output_channels:
                            output_i_real += coef*tf.einsum('lmk,nkpi,mpi->nlpi', (self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1  ] )
                            output_i_real +=-coef*tf.einsum('lmk,nkpi,mpi->nlpi', (self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,nkpi,mpi->nlpi', (self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,nkpi,mpi->nlpi', (self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1  ] )
                        else:
                            output_i_real += coef*tf.einsum('lmk,nkpi,mpij->nlpj', (self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1  ] )
                            output_i_real +=-coef*tf.einsum('lmk,nkpi,mpij->nlpj', (self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,nkpi,mpij->nlpj', (self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2  ], self.weights_ch_list[2*l1+1] )
                            output_i_imag += coef*tf.einsum('lmk,nkpi,mpij->nlpj', (self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2]), input[2*l2+1], self.weights_ch_list[2*l1  ] )
                    itr +=1

            output.append(output_i_real + self.biases_list[2*l])
            output.append(output_i_imag + self.biases_list[2*l+1])
                
            
        return output + self.biases_list



class FunctionNormalization(tf.keras.layers.Layer):
    def __init__(self, num_degrees = 4, num_radii = 5, name = 'normalization', l1_norm = True):
        super(FunctionNormalization, self).__init__()
        self.L = num_degrees
        self.name_ch = name
        self.P = num_radii
        self.l1_norm = l1_norm
        w = np.ones((1,1, self.P, 1))
        w[:,0,0,:] = 0.0
        self.w_tensor = tf.convert_to_tensor(w, dtype = tf.float32, name = name + 'extract_first_coef')
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.built = True
    def get_output_shapes(self):
        return [tf.TensorShape((None, l//2, self.P,  self.input_channels)) for l in range(2*self.L)] 
    def compute_output_shape(self, input_shape):
        return [(None, l//2, self.P,  self.input_channels) for l in range(2*self.L)] 
    def get_config(self):
        config = super(FunctionNormalization, self).get_config()
        config.update({"L": self.L})
        config.update({"name_ch": self.name_ch})
        config.update({"P": self.P})
        config.update({"w_tensor": self.w_tensor})

        return config
    def call(self, input):
        self.gamma = input[-1]
        self.beta = input[-2]
        input = input[:-2]
        if self.l1_norm:
            input[0] = tf.multiply(self.w_tensor, input[0])
        sum_sq_value  = tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(input[0]), axis = 2), axis = 1),[-1,1,1,self.input_channels])
        sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(input[1]), axis = 2), axis = 1),[-1,1,1,self.input_channels])
        for l in range(1, self.L):
            sum_sq_value += 2*tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(input[2*l  ]), axis = 2), axis = 1),[-1,1,1,self.input_channels])
            sum_sq_value += 2*tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(input[2*l+1]), axis = 2), axis = 1),[-1,1,1,self.input_channels])
        var = sum_sq_value/(self.L**2)
        output = []
        for l in range(self.L):
            output.append(self.gamma*(input[2*l  ])/tf.sqrt(var+EPS) + self.beta)
            output.append(self.gamma*(input[2*l+1])/tf.sqrt(var+EPS) + self.beta)
        return output


class FunctionActivation(tf.keras.layers.Layer):
    def __init__(self, num_degrees = 4, num_radii = 5, name = 'activation'):
        super(FunctionActivation, self).__init__()
        self.L = num_degrees
        self.P = num_radii 
        self.name_ch = name
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.built = True
    def get_output_shapes(self):
        return [tf.TensorShape((None, l//2, self.P,  self.input_channels)) for l in range(2*self.L)] 
    def compute_output_shape(self, input_shape):
        return [(None, l//2, self.P,  self.input_channels) for l in range(2*self.L)] 
    def get_config(self):
        config = super(FunctionActivation, self).get_config()
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"name_ch": self.name_ch})
        return config
    def call(self, input):
        condition = 0.25*tf.math.accumulate_n([(1 + 1*(l>0))*tf.reduce_sum(tf.reduce_mean(tf.square(input[l]- input[2*self.L + l]), axis = 2), axis = 1) for l in range(2*self.L)])/(self.L**2)
        with tf.name_scope(self.name_ch + "_condition" ):
            tf.summary.histogram(self.name_ch + "_condition", condition)
        condition = tf.reshape(condition, [-1, 1, 1, self.input_channels])
        
        output = [(LEAK + (1 - LEAK)*condition)*(input[l] ) for l in range(2*self.L)]
        return output


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, if_aggregation=True, output_channels=10, num_degrees = 4, num_radii = 5, num_eret = 15, name = "ConvolutionBlock", use_aggregation_tensors = True, l1_norm = True, normalization_bool = True, shift_frames =True, multiplication = False, use_edge_retyping_in_aggregation = True, use_diagonal_filter_if_possible = False):
        super(ConvolutionBlock, self).__init__()
        self.name_ch = name
        self.if_aggregation = if_aggregation
        self.L = num_degrees
        self.T = num_eret
        self.P = num_radii
        self.output_channels = output_channels
        self.normalization_bool = normalization_bool
        self.use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation 
        self.use_diagonal_filter_if_possible = use_diagonal_filter_if_possible
        self.shift_frames = shift_frames
        self.multiplication = multiplication
        if self.if_aggregation:
            self.Aggregation = NodesAggregation(num_degrees=self.L, num_radii = self.P, num_eret=self.T, name = name + '_aggregation', shift_frames = self.shift_frames, multiplication = self.multiplication, use_edge_retyping_in_aggregation = self.use_edge_retyping_in_aggregation) 
        self.Convolution = Convolution6D(name = name + '_convolution_6D', output_channels=self.output_channels, num_degrees = self.L, num_radii = self.P, num_eret=self.T, after_aggregation=self.if_aggregation, use_diagonal_filter_if_possible = self.use_diagonal_filter_if_possible, use_edge_retyping_in_aggregation = self.use_edge_retyping_in_aggregation)
        if self.normalization_bool:
            self.Normalization = FunctionNormalization(num_degrees = self.L, num_radii = self.P , name = name + '_normalization', l1_norm = l1_norm)
        self.Activation = FunctionActivation(num_degrees=self.L, num_radii = self.P,  name = name + '_activation' )
        
    
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        current_input_shapes = input_shapes
        if self.if_aggregation:
            self.Aggregation.build(current_input_shapes)
            current_input_shapes = self.Aggregation.get_output_shapes()
        self.Convolution.build(current_input_shapes)
        current_input_shapes = self.Convolution.get_output_shapes()
        if self.normalization_bool:
            self.gamma = tf.Variable(1.0, trainable=True, name=self.name_ch+"_normalization_gamma")
            self.beta = tf.Variable(0.01, trainable=True, name=self.name_ch+"_normalization_beta")
            self.Normalization.build(current_input_shapes)
            current_input_shapes = self.Normalization.get_output_shapes()
        self.Activation.build(current_input_shapes)
        self.output_shapes = self.Activation.get_output_shapes()
        self.built = True
        
    def get_output_shapes(self):
        return self.output_shapes

    def compute_output_shape(self, input_shape):
        return self.Activation.compute_output_shape(input_shape)
    
    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update({"name_ch": self.name_ch})
        config.update({"if_aggregation": self.if_aggregation})
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"T": self.T})
        config.update({"output_channels": self.output_channels})
        if self.if_aggregation:
            config.update({"Aggregation": self.Aggregation})
        config.update({"Convolution": self.Convolution})
        if self.normalization_bool:
            config.update({"Normalization": self.Normalization})
        config.update({"Activation": self.Activation})
        return config
    def call(self, input):
        if self.if_aggregation:
            nodes_start = self.Aggregation(input)
        else:
            nodes_start = input[:2*self.L]
        nodes_and_biases = self.Convolution(nodes_start)
        nodes, biases = nodes_and_biases[:2*self.L], nodes_and_biases[2*self.L:]
        if self.normalization_bool:
            nodes = self.Normalization(nodes + [self.beta, self.gamma])
            biases = self.Normalization(biases + [self.beta, self.gamma])
        nodes = self.Activation(nodes+biases)

                
        
        return nodes
        
        





class FunctionToValue(tf.keras.layers.Layer):
    def __init__(self, num_degrees = 4, num_radii = 5, name = 'fun_to_vec'):
        super(FunctionToValue, self).__init__()
        self.name_ch = name
        self.L = num_degrees
        self.P = num_radii
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.weights_ch = []
        for l in range(self.L):
            weights_degree_real = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_real", shape = [1, l+1, self.P, self.input_channels], initializer = 'truncated_normal')
            weights_degree_imag = self.add_weight("weights_" + self.name_ch + "_" + str(l) + "_imag", shape = [1, l+1, self.P, self.input_channels], initializer = 'truncated_normal')
            self.weights_ch.append(weights_degree_real)
            self.weights_ch.append(weights_degree_imag)
        self.built = True
    def get_output_shapes(self):
        return tf.TensorShape((None, self.input_channels)) 
    
    def compute_output_shape(self, input_shape):
        return (None, self.input_channels)

    def get_config(self):
        config = super(FunctionToValue, self).get_config()
        config.update({"name_ch": self.name_ch})
        config.update({"L": self.L})
        config.update({"P": self.P})
        return config
    def call(self, input):
        output = tf.math.accumulate_n([tf.reduce_mean(tf.reduce_mean(self.weights_ch[l]*input[l], axis = 2), axis = 1) for l in range(2*self.L)])
        return output



class FunctionalBlock(tf.keras.layers.Layer):
    def __init__(self, layers_params = ['Conv_30', 'ConvAgg_40', 'ConvAgg_30'], name = 'FunctionalBlock', num_degrees = 5, num_radii = 5, num_eret = 15, beta = 0.05, use_aggregation_tensors = True, l1_norm = True, normalization_bool = True, use_edge_retyping_in_aggregation = True, use_diagonal_filter_if_possible = False,  shift_frames = False):
        super(FunctionalBlock, self).__init__()
        self.L = num_degrees
        self.P = num_radii
        self.T = num_eret 
        self.Dropout = Dropout_coefficients(beta= beta, name = name + '_dropout') 
        self.Layers = []
        self.normalization_bool = normalization_bool
        self.use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation
        self.use_diagonal_filter_if_possible = use_diagonal_filter_if_possible
        self.shift_frames = shift_frames 
        for l_i, layer in enumerate(layers_params):
            if layer.split('_')[0] == 'ConvAgg':
                output_channels = int(layer.split('_')[1])
                self.Layers.append(ConvolutionBlock(if_aggregation=True, output_channels=output_channels, num_degrees=self.L, num_radii = self.P, num_eret=self.T, name = name + "_" + str(l_i) + '_ConvAgg', use_aggregation_tensors = use_aggregation_tensors, normalization_bool = self.normalization_bool, l1_norm = l1_norm, shift_frames = self.shift_frames, use_edge_retyping_in_aggregation = self.use_edge_retyping_in_aggregation, use_diagonal_filter_if_possible = self.use_diagonal_filter_if_possible))
            elif layer.split('_')[0] == 'ConvMult':
                output_channels = int(layer.split('_')[1])
                self.Layers.append(ConvolutionBlock(if_aggregation=True, output_channels=output_channels, num_degrees=self.L, num_radii = self.P, num_eret=self.T, name = name + "_" + str(l_i) + '_ConvMult', use_aggregation_tensors = use_aggregation_tensors, normalization_bool = self.normalization_bool, l1_norm = l1_norm, multiplication = True, use_edge_retyping_in_aggregation = self.use_edge_retyping_in_aggregation, use_diagonal_filter_if_possible = self.use_diagonal_filter_if_possible))
            elif layer.split('_')[0] == 'Conv':
                output_channels = int(layer.split('_')[1])
                self.Layers.append(ConvolutionBlock(if_aggregation=False, output_channels=output_channels, num_degrees=self.L, num_radii = self.P, num_eret=self.T, name = name + "_" + str(l_i) + '_Conv', use_aggregation_tensors = use_aggregation_tensors, normalization_bool = self.normalization_bool, l1_norm = l1_norm, use_diagonal_filter_if_possible = self.use_diagonal_filter_if_possible))
            
        self.FunToVector = FunctionToValue(num_degrees = self.L, num_radii = self.P, name = name + '_FunToVec')

    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        current_input_shapes = input_shapes
        for l_i in range(len(self.Layers)):
            self.Layers[l_i].build(current_input_shapes)
            current_input_shapes = self.Layers[l_i].get_output_shapes() +input_shapes[2*self.L:]
        self.FunToVector.build(current_input_shapes)
        self.output_shapes = self.FunToVector.get_output_shapes()
        self.built = True
        
    def get_output_shapes(self):
        return self.output_shapes

    def compute_output_shape(self, input_shape):
        return self.FunToVector.compute_output_shape(input_shape)
        
    def get_config(self):
        config = super(FunctionalBlock, self).get_config()
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"T": self.T})
        for i_l, Layer in enumerate(self.Layers):
             config.update({"Layer_"+str(i_l): Layer})
        return config
    
    def call(self, input, training = None):
        if len(self.Layers) == 0:
            nodes = input[:2*self.L]
        for l_i in range(len(self.Layers)):
            nodes = self.Layers[l_i](input)
            nodes = self.Dropout(nodes, training = training)
            input = nodes + input[2*self.L:]
        nodes  = self.FunToVector(nodes)
        return nodes





class ClassicalGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, num_eret = 15,  num_degrees = 5,  output_channels = 10, activation = 'lrelu', name = 'graph_conv', use_bias = False, add_sph_nodes = False):
        super(ClassicalGraphConvolution, self).__init__()
        self.T = num_eret
        self.L = num_degrees
        self.output_channels = output_channels
        self.activation = activation
        self.name_ch = name
        self.use_bias = use_bias
        self.add_sph_nodes = add_sph_nodes
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        if self.add_sph_nodes:
            self.weights_ch = self.add_weight("weights_" + self.name_ch , shape = [self.input_channels, self.output_channels, self.L**2, self.T], initializer = 'truncated_normal')  
        else:
            self.weights_ch = self.add_weight("weights_" + self.name_ch , shape = [self.input_channels, self.output_channels,  self.T], initializer = 'truncated_normal')  
        self.weights_ch2 = self.add_weight("weights_" + self.name_ch + "_2", shape = [self.input_channels, self.output_channels], initializer = 'truncated_normal')
        
        if self.use_bias:
            self.bias_ch = self.add_weight("bias_" + self.name_ch , shape = [self.output_channels], initializer = 'truncated_normal') 
        
        self.built = True
        
    def get_output_shapes(self):
        return tf.TensorShape((None, self.output_channels)) 

    def compute_output_shape(self, input_shape):
        return (None, self.output_channels) 

    def get_config(self):
        config = super(ClassicalGraphConvolution, self).get_config()
        config.update({"T": self.T})
        config.update({"output_channels": self.output_channels})
        config.update({"activation": self.activation})
        config.update({"name_ch": self.name_ch})
        config.update({"use_bias": self.use_bias})
        return config
        
        
    def call(self, input_array):
        input = input_array[0]
        adjacency_matrix = input_array[1]
        if self.add_sph_nodes:
            temp = tf.einsum('mnlt,ni,ijlt->mj', adjacency_matrix, input, self.weights_ch)
        else:
            temp = tf.einsum('mnt,ni,ijt->mj', adjacency_matrix, input, self.weights_ch)
        temp += tf.einsum('ni,ij->nj', input, self.weights_ch2)
        if self.use_bias:
            temp += self.bias_ch
        if self.activation == 'lrelu':
            output = lrelu(temp, leak = LEAK)
        elif self.activation == 'tanh':
            output = tf.tanh(temp)
        else:
            output = temp
        return output



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_eret = 15, output_channels = 10, hidden_space_dims = 10, activation = 'lrelu', name = 'graph_conv', use_bias = False, second_order = False):
        super(MultiHeadAttention, self).__init__()
        self.T = num_eret
        self.output_channels = output_channels
        self.H = hidden_space_dims
        self.activation = activation
        self.name_ch = name
        self.use_bias = use_bias
        self.second_order = second_order
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        self.weights_ch1 = self.add_weight("weights_" + self.name_ch + "_1", shape = [2*self.input_channels, (self.T+1), self.H], initializer = 'truncated_normal')  
        self.weights_ch2 = self.add_weight("weights_" + self.name_ch + "_2", shape = [2*self.input_channels, (self.T+1), self.output_channels], initializer = 'truncated_normal')
        self.weights_ch3 = self.add_weight("weights_" + self.name_ch + "_3", shape = [self.input_channels, self.H], initializer = 'truncated_normal')
        if self.second_order:
            self.weights_ch4 = self.add_weight("weights_" + self.name_ch + "_4", shape = [2*self.input_channels, (self.T+1), self.H], initializer = 'truncated_normal')
            self.weights_ch5 = self.add_weight("weights_" + self.name_ch + "_5", shape = [2*self.input_channels, (self.T+1), self.output_channels], initializer = 'truncated_normal')
            self.weights_ch6 = self.add_weight("weights_" + self.name_ch + "_6", shape = [self.input_channels, self.H], initializer = 'truncated_normal')
        if self.use_bias:
            self.bias_ch1 = self.add_weight("bias_" + self.name_ch + '_1' , shape = [1,1,self.H], initializer = 'truncated_normal')
            self.bias_ch2 = self.add_weight("bias_" + self.name_ch + '_2', shape = [1,self.output_channels], initializer = 'truncated_normal')
            self.bias_ch3 = self.add_weight("bias_" + self.name_ch + '_3', shape = [1,self.H], initializer = 'truncated_normal') 
            if self.second_order:
                self.bias_ch4 = self.add_weight("bias_" + self.name_ch + '_4' , shape = [1,1,self.H], initializer = 'truncated_normal')
                self.bias_ch6 = self.add_weight("bias_" + self.name_ch + '_6', shape = [1,self.H], initializer = 'truncated_normal')       
 
        self.built = True
        
    def get_output_shapes(self):
        return tf.TensorShape((None, self.output_channels)) 

    def compute_output_shape(self, input_shape):
        return (None, self.output_channels) 

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"T": self.T})
        config.update({"H": self.H})
        config.update({"output_channels": self.output_channels})
        config.update({"activation": self.activation})
        config.update({"name_ch": self.name_ch})
        config.update({"use_bias": self.use_bias})
        return config
        
        
    def call(self, input_array):
        input = input_array[0]
        adjacency_matrix = input_array[1]
        input_diag = tf.expand_dims(tf.transpose(tf.linalg.diag(tf.transpose(input, [1,0])), [1,2,0]), axis = 3)
        D = tf.concat([tf.einsum('mnt,ni->mnit', adjacency_matrix, input), input_diag], 3)
        D = tf.concat([D,tf.transpose(D, [1,0,2,3])],axis = 2)
        # D = tf.reshape(D, [-1, 2*self.input_channels* (self.T+1)])
        
        K = tf.einsum('mnit,ith->mnh', D,  self.weights_ch1)
        V = tf.einsum('mnit,itd->mnd', D,  self.weights_ch2)
        Q = tf.einsum('ni,ih->nh', input,  self.weights_ch3)

        if self.use_bias:
            K += self.bias_ch1
            Q += self.bias_ch3

        temp1 = tf.keras.activations.softmax(tf.einsum('nh,nmh->nm', Q,  K), axis = [1] )
        temp2 = tf.keras.activations.softmax(tf.einsum('mh,nmh->nm', Q,  K), axis = [0] ) 

        temp = tf.einsum( 'nm,nmd->nd'  ,temp1 , V) + tf.einsum( 'nm,nmd->md'  ,temp2 , V)
        if self.second_order:
            adjacency_matrix_so = tf.einsum('mnt,nkt->mkt',adjacency_matrix, adjacency_matrix)
            D = tf.concat([tf.einsum('mnt,ni->mnit', adjacency_matrix_so, input), input_diag], 3)
            D = tf.concat([D,tf.transpose(D, [1,0,2,3])],axis = 2)

            K = tf.einsum('mnit,ith->mnh', D,  self.weights_ch4)
            V = tf.einsum('mnit,itd->mnd', D,  self.weights_ch5)
            Q = tf.einsum('ni,ih->nh', input,  self.weights_ch6)

            if self.use_bias:
                K += self.bias_ch4
                Q += self.bias_ch6

            temp3 = tf.keras.activations.softmax(tf.einsum('nh,nmh->nm', Q,  K), axis = [1] )
            temp4 = tf.keras.activations.softmax(tf.einsum('mh,nmh->nm', Q,  K), axis = [0] )

            temp += tf.einsum( 'nm,nmd->nd'  ,temp3 , V) + tf.einsum( 'nm,nmd->md'  ,temp4 , V)
        if self.use_bias:
            temp += self.bias_ch2
        if self.activation == 'lrelu':
            output = lrelu(temp, leak = LEAK)
        elif self.activation == 'tanh':
            output = tf.tanh(temp)
        else:
            output = temp
        return output


class VectorBlock(tf.keras.layers.Layer):
    def __init__(self, num_eret = 15, num_degrees = 5, output_channels_list = [20,12,5], activation = 'lrelu', name = 'VectorBlock', use_bias = False, mha = False, second_order = False, add_sph_nodes = False):
        super(VectorBlock, self).__init__()
        self.T = num_eret
        self.output_channels_list = output_channels_list
        self.activation = activation
        self.name_ch = name
        self.use_bias = use_bias
        self.mha = mha
        self.second_order = second_order
        self.add_sph_nodes = add_sph_nodes
        self.L = num_degrees
        self.Layers = []
        for o_i, output_channels in enumerate(output_channels_list):
            if self.mha:
                self.Layers.append(MultiHeadAttention(num_eret = self.T, output_channels = output_channels, hidden_space_dims = output_channels, activation = self.activation, use_bias = self.use_bias, second_order = self.second_order, name = name+'_MHA_'+str(o_i)))
            else:
                self.Layers.append(ClassicalGraphConvolution(num_eret = self.T, num_degrees = self.L, output_channels = output_channels, activation = self.activation, use_bias = self.use_bias, add_sph_nodes = self.add_sph_nodes, name = name+'_GraphConv_'+str(o_i)))

    
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        current_input_shapes = input_shapes
        for l_i in range(len(self.Layers)):
            self.Layers[l_i].build(current_input_shapes)
            current_input_shapes = [self.Layers[l_i].get_output_shapes()] + [tf.TensorShape((None, None, self.T))] 
        if len(self.Layers) == 0:
            self.output_shapes = input_shapes[0]
        else:
            self.output_shapes = self.Layers[-1].get_output_shapes()
        self.built = True
        # super(VectorBlock, self).build(input_shapes)
    
    def get_output_shapes(self):
        return self.output_shapes

    def compute_output_shape(self, input_shape):
        if len(self.Layers) == 0:
            return input_shape[0]
        return  self.Layers[-1].compute_output_shape(input_shape)

    def get_config(self):
        config = super(VectorBlock, self).get_config()
        config.update({"T": self.T})
        config.update({"output_channels_list": self.output_channels_list})
        config.update({"activation": self.activation})
        config.update({"name_ch": self.name_ch})
        config.update({"use_bias": self.use_bias})
        for i_l, Layer in enumerate(self.Layers):
            config.update({"Layer_"+str(i_l): Layer})
        return config
    
    def call(self, input_array):
        if len(self.Layers) == 0:
            nodes = input_array[0]
        for l_i in range(len(self.Layers)):
            nodes = self.Layers[l_i](input_array)
            input_array = [nodes] + [input_array[1]]
        return nodes

class MuOrStdFromDistribution(tf.keras.layers.Layer):
    def __init__(self, name = 'MuOrStdFromDistribution', use_bias = True, hidden_size = 3, activation = 'lrelu'):
        super(MuOrStdFromDistribution, self).__init__()
        self.use_bias = use_bias
        self.name_ch = name
        self.hidden_size = hidden_size
        self.activation = activation


    def build(self, input_shape):
        self.input_channels = input_shape.as_list()[-1]
        self.weights_mutohdn = self.add_weight("weights_" + self.name_ch + "_mutohdn", shape = [self.input_channels, self.hidden_size], initializer = 'truncated_normal') 
        self.weights_covtohdn = self.add_weight("weights_" + self.name_ch + "_covtohdn", shape = [self.input_channels, self.input_channels, self.hidden_size], initializer = 'truncated_normal') 
        self.weights_muhdntoout = self.add_weight("weights_" + self.name_ch + "_muhdntoout", shape = [self.hidden_size, 1], initializer = 'truncated_normal')
        self.weights_covhdntoout = self.add_weight("weights_" + self.name_ch + "_covhdntoout", shape = [self.hidden_size, 1], initializer = 'truncated_normal')
        if self.use_bias:
            self.bias_mu = self.add_weight("weights_" + self.name_ch + "_biasmu", shape = [self.hidden_size], initializer = 'truncated_normal')
            self.bias_cov = self.add_weight("weights_" + self.name_ch + "_biascov", shape = [self.hidden_size], initializer = 'truncated_normal')

        self.built = True

    def get_output_shapes(self):
        return tf.TensorShape((1)) 

    def compute_output_shape(self, input_shape):
        return (1) 


    def get_config(self):
        config = super(MuOrStdFromDistribution, self).get_config()
        config.update({"hidden_size": self.hidden_size})
        config.update({"name_ch": self.name_ch})
        config.update({"use_bias": self.use_bias})
        config.update({"activation": self.activation})
        return config
    

    def call(self, input_tensor):
        mu_tensor = tf.math.reduce_mean(input_tensor, axis = 0, keepdims=True)
        tensor_shifted = input_tensor - mu_tensor
        mu_tensor = tf.reshape(mu_tensor, [self.input_channels])
        cov_tensor = tf.reduce_mean(tf.einsum('ne,nt->net', tensor_shifted, tensor_shifted ), axis = 0)
        mu_hidden = tf.einsum('t,th->h', mu_tensor, self.weights_mutohdn )
        cov_hidden = tf.einsum('et,eth->h', cov_tensor, self.weights_covtohdn )
        if self.use_bias:
            mu_hidden += self.bias_mu
            cov_hidden += self.bias_cov
        if self.activation == 'lrelu':
            mu_hidden = lrelu(mu_hidden, leak = LEAK)
            cov_hidden = lrelu(cov_hidden, leak = LEAK)
        elif self.activation == 'tanh':
            mu_hidden = tf.tanh(mu_hidden)
            cov_hidden = tf.tanh(cov_hidden)
        return tf.einsum('h,ho->o', mu_hidden, self.weights_muhdntoout ) + tf.einsum('h,ho->o', cov_hidden, self.weights_covhdntoout )



class FunctionalGraphNetwork(tf.keras.Model):
    def __init__(self,  name = 'FunctionalGraphNetwork', num_degrees = 5, num_radii = 5, num_et = 403, num_eret = 15, model_params = None, ns_sample = 10, beta = 0.05, last_channels = 1, retype_dims = 15 , activation = 'lrelu', task = 'QA', use_bias = True, mha = False, apply_gradients = 40, use_aggregation_tensors = True, add_sph_nodes = False, shift_frames = True, find_mean_std = False, non_linear_edge_retyper = False, non_linear_atom_retyper = False, nlar_second_order = False, normalization_bool = True, use_edge_retyping_in_aggregation = True,  use_diagonal_filter_if_possible = False):
        super(FunctionalGraphNetwork, self).__init__()
        self.L = num_degrees
        self.P = num_radii
        self.T = num_eret
        self.E = num_et
        self.retype_dims = retype_dims
        self.model_params = model_params
        self.last_channels = last_channels
        self.activation = activation
        self.task = task
        self.NS = ns_sample
        self.use_bias = use_bias
        self.mha = mha 
        self.second_order = mha
        self.beta = beta
        self.apply_gradient_each = apply_gradients
        self.training_step_iter = 0
        self.use_aggregation_tensors = use_aggregation_tensors
        self.add_sph_nodes = add_sph_nodes
        self.shift_frames = shift_frames
        self.l1_norm = True
        self.normalization_bool = normalization_bool
        self.find_mean_std = find_mean_std
        self.non_linear_edge_retyper = non_linear_edge_retyper
        self.non_linear_atom_retyper = non_linear_atom_retyper
        self.nlar_second_order = nlar_second_order
        self.use_edge_retyping_in_aggregation = use_edge_retyping_in_aggregation
        self.use_diagonal_filter_if_possible = use_diagonal_filter_if_possible
        self.Retyper = Retyper(output_channels = self.retype_dims, name = name + '_retyper', num_degrees = self.L, num_radii = self.P, non_linear_atom_retyper = self.non_linear_atom_retyper, activation = 'tanh', second_layer=self.nlar_second_order)
        self.Functional_part = FunctionalBlock(layers_params = self.model_params.layers, name = name + '_FunctionalBlock', num_degrees = self.L, num_radii = self.P, num_eret = self.T, beta = self.beta, use_aggregation_tensors = self.use_aggregation_tensors, l1_norm = self.l1_norm, normalization_bool = self.normalization_bool, use_edge_retyping_in_aggregation = self.use_edge_retyping_in_aggregation, use_diagonal_filter_if_possible = self.use_diagonal_filter_if_possible,  shift_frames = self.shift_frames)
        self.Vector_part = VectorBlock(num_eret = self.T, num_degrees = self.L, output_channels_list = self.model_params.output_channels, activation = self.activation, name = name + '_VectorBlock', use_bias = self.use_bias, mha = self.mha, second_order = self.second_order, add_sph_nodes = self.add_sph_nodes)
        
        
        if self.task == 'QA':
            self.last_layer = ClassicalGraphConvolution(num_eret = self.T, output_channels = 1, activation = 'tanh', name = name + '_FinalGraphConv_' + task)
            if self.find_mean_std:
                self.mu_layer = MuOrStdFromDistribution(name = 'Mu_layer', use_bias = use_bias, hidden_size = self.model_params.output_channels[-1], activation = 'tanh')
                self.std_layer = MuOrStdFromDistribution(name = 'Std_layer', use_bias = use_bias, hidden_size = self.model_params.output_channels[-1], activation = 'tanh')
            else:
                self.bias_QA = tf.Variable(0.5, name = "bias_QA", trainable = True)
                self.dev_QA = tf.Variable(1.0, name = "dev_QA", trainable = True)
        elif self.task == 'refinement':
            self.last_layer = ClassicalGraphConvolution(num_eret = self.T, output_channels = 3, activation = 'tanh', name = name + '_FinalGraphConv_' + task)
        self.accumulated_gradients = 0#[tf.zeros_like(tv) for tv in self.trainable_variables]
    
    def build(self, input_shapes):
        self.input_channels = input_shapes[0].as_list()[-1]
        current_input_shapes = input_shapes
        self.Retyper.build(current_input_shapes)
        current_input_shapes = self.Retyper.get_output_shapes()
        current_input_shapes += [tf.TensorShape((None, None)) for g in range(2)]
        current_input_shapes += [tf.TensorShape((None, self.T)) ]
        current_input_shapes += [tf.TensorShape((None, l//4+1, l//4+1)) for l in range(4*self.L)]
        current_input_shapes += [tf.TensorShape((None, self.P, (l//2%(self.L))+1, (l//2//(self.L))+1)) for l in range(2*self.L**2)]
        current_input_shapes += [tf.TensorShape((None, l//4+1, l//4+1)) for l in range(4*self.L)]
        self.Functional_part.build(current_input_shapes)
        current_input_shapes = [self.Functional_part.get_output_shapes()] + [tf.TensorShape((None, None, self.T)) ] 
        self.Vector_part.build(current_input_shapes)
        current_input_shapes = [self.Vector_part.get_output_shapes()] + [tf.TensorShape((None, None, self.T)) ] 
        self.last_layer.build(current_input_shapes)
        self.output_shapes = self.last_layer.get_output_shapes()
        if self.non_linear_edge_retyper:
            self.weights_ch1 = self.add_weight("edges_retyper_1" , shape = [self.E, (self.T+self.E)//2], initializer = 'truncated_normal')
            self.weights_ch2 = self.add_weight("edges_retyper_2" , shape = [(self.T+self.E)//2, self.T], initializer = 'truncated_normal')
            self.bias_ch = self.add_weight("edges_retyper_bias" , shape = [(self.T+self.E)//2], initializer = 'truncated_normal')
        else:
            self.weights_ch = self.add_weight("edges_retyper_" , shape = [self.E, self.T], initializer = 'truncated_normal')
        if self.find_mean_std:
            self.mu_layer.build(current_input_shapes[0])
            self.std_layer.build(current_input_shapes[0])
    
    
    def compute_output_shape(self, input_shapes):
        num_nodes = input_shapes[0].as_list()[1]
        return  (1, num_nodes, 1)

    def get_config(self):
        config = super(FunctionalGraphNetwork, self).get_config()
        config.update({"L": self.L})
        config.update({"P": self.P})
        config.update({"T": self.T})
        config.update({"retype_dims": self.retype_dims})
        config.update({"model_params": self.model_params})
        config.update({"last_channels": self.last_channels})
        config.update({"activation": self.activation})
        config.update({"task": self.task})
        config.update({"Retyper": self.Retyper})
        config.update({"Functional_part": self.Functional_part})
        config.update({"Vector_part": self.Vector_part})
        config.update({"last_layer": self.last_layer})
        if self.task == 'QA':
            config.update({"bias_QA": self.bias_QA})
            config.update({"dev_QA": self.dev_QA})


        return config
    
    
    def call(self, input_array_raw, training = None):
        input_array = []
        # print(tf.shape(input_array_raw[0]))
        for l in range(len(input_array_raw)):
            if (l+3)%(self.NS+2)==0:
                input_array.append(tf.sparse.reorder(tf.sparse.SparseTensor(indices = tf.reduce_sum(input_array_raw[l], axis = 0), values = tf.reduce_sum(tf.reduce_sum(input_array_raw[l+1], axis = 0), axis =1), dense_shape = tf.reduce_sum(input_array_raw[l+2], axis = 0))))
            elif (l+2)%(self.NS+2)==0 or  (l+1)%(self.NS+2)==0:
                continue 
            else:
                input_array.append(tf.reduce_sum(input_array_raw[l], axis = 0))
        B = len(input_array)//self.NS
        outputs = []
        outputs_global = []
        for i_b in range(B):
            input_array_i = input_array[i_b*self.NS: (i_b+1)*self.NS]
            if self.non_linear_edge_retyper:
                if self.use_aggregation_tensors:
                    input_array_i[2*self.L+2] = tf.einsum('ey,yt->et', input_array_i[2*self.L+2], self.weights_ch1)
                    input_array_i[2*self.L+2] = tf.tanh(input_array_i[2*self.L+2] + self.bias_ch)
                    input_array_i[2*self.L+2] = tf.einsum('ey,yt->et',   input_array_i[2*self.L+2], self.weights_ch2) 
                if self.add_sph_nodes:
                    input_array_i[-1]         = tf.einsum('mnly,yt->mnlt', tf.sparse.to_dense(input_array_i[-1], validate_indices=False), self.weights_ch1)
                    input_array_i[-1]         = tf.tanh(input_array_i[-1]         + self.bias_ch)
                    input_array_i[-1]         = tf.einsum('mnly,yt->mnlt', input_array_i[-1],         self.weights_ch2) 
                else:
                    input_array_i[-1]         = tf.einsum('mny,yt->mnt', tf.sparse.to_dense(input_array_i[-1], validate_indices=False), self.weights_ch1)
                    input_array_i[-1]         = tf.tanh(input_array_i[-1]         + self.bias_ch)
                    input_array_i[-1]         = tf.einsum('mny,yt->mnt', input_array_i[-1],         self.weights_ch2) 
                 
                
                
            else:
                if self.use_aggregation_tensors:
                    input_array_i[2*self.L+2] = tf.einsum('ey,yt->et', input_array_i[2*self.L+2], self.weights_ch)
                if self.add_sph_nodes:
                    input_array_i[-1] = tf.einsum('mnly,yt->mnlt', tf.sparse.to_dense(input_array_i[-1], validate_indices=False), self.weights_ch)
                else:
                    input_array_i[-1] = tf.einsum('mny,yt->mnt', tf.sparse.to_dense(input_array_i[-1], validate_indices=False), self.weights_ch)
            input_ret = input_array_i[:2*self.L]
            nodes = self.Retyper(input_ret)
            input_fun_part = nodes + input_array_i[2*self.L:-1]
            nodes = self.Functional_part(input_fun_part, training = training)
            input_vector_part = [nodes] + input_array_i[-1:]
            nodes = self.Vector_part(input_vector_part)
            input_final_part = [nodes] + input_array_i[-1:]
            nodes = self.last_layer(input_final_part)
            if self.task == 'QA':
                if self.find_mean_std:
                    self.bias_QA = self.mu_layer(input_final_part[0]) + 0.5
                    self.dev_QA = self.std_layer(input_final_part[0]) + 1.0
                nodes = self.dev_QA*nodes + self.bias_QA
                nodes = tf.reshape(nodes, [1,-1,1])
            else:
                nodes = tf.reshape(nodes, [1,-1,3])
            outputs_global.append(tf.reshape(tf.reduce_mean(nodes, 1),[1,1,1]))
            outputs.append(nodes)
        
        output = tf.concat(outputs, 1, name = 'output_1')
        output_global = tf.concat(outputs_global, 1, name = 'output_2')
        output_dict = {'output_1': output, 'output_2': output_global}
        return output_dict
    def reset_gradients(self):
        self.accumulated_gradients = [tf.zeros_like(tv) for tv in self.trainable_variables]
 
    def train_step(self, data):
        self.training_step_iter += 1
        print(self.training_step_iter)
        if self.training_step_iter == 1:
            self.accumulated_gradients = [tf.zeros_like(tv) for tv in self.trainable_variables]
            self.gt_bs = tf.zeros([1, 0, 1], tf.float32)
            self.pr_bs = tf.zeros([1, 0, 1], tf.float32)
            self.gt_gl_bs = tf.zeros([1, 0, 1], tf.float32)
            self.pr_gl_bs = tf.zeros([1, 0, 1], tf.float32)
        x, y = data
        # self.gt_bs = tf.concat([self.gt_bs, y], 1)
        self.gt_bs = tf.concat([self.gt_bs, y['output_1']], 1)
        self.gt_gl_bs = tf.concat([self.gt_bs, y['output_2']], 1)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) 
            # self.pr_bs = tf.concat([self.pr_bs, y_pred], 1)
            self.pr_bs = tf.concat([self.pr_bs, y_pred['output_1']], 1) 
            self.pr_gl_bs = tf.concat([self.pr_bs, y_pred['output_2']], 1)
            loss_value = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradient_i = tape.gradient(loss_value, trainable_vars)
        self.accumulated_gradients = [(acum_grad+grad) for acum_grad, grad in zip(self.accumulated_gradients, gradient_i)]
        if self.training_step_iter%self.apply_gradient_each == 0:
            self.optimizer.apply_gradients(zip(self.accumulated_gradients/self.apply_gradient_each, trainable_vars))
            self.accumulated_gradients = [tf.zeros_like(tv) for tv in self.trainable_variables]
            self.gt_bs_dict = {'output_1': self.gt_bs, 'output_2': self.gt_gl_bs}
            self.pr_bs_dict = {'output_1': self.pr_bs, 'output_2': self.pr_gl_bs}
            self.compiled_metrics.update_state(self.gt_bs_dict, self.pr_bs_dict)
            self.gt_bs = tf.zeros([1, 0, 1], tf.float32)
            self.pr_bs = tf.zeros([1, 0, 1], tf.float32)
            self.gt_gl_bs = tf.zeros([1, 0, 1], tf.float32)
            self.pr_gl_bs = tf.zeros([1, 0, 1], tf.float32)
        output = {m.name: m.result() for m in self.metrics}
        output['prediction_global'] = self.pr_gl_bs
        tf.keras.print_tensor(self.pr_gl_bs, "Prediction (y_pred) =")
        #output['output_1_pearson_correlation'] = -1
        return output

    # def test_step(self, data):
    #    x, y = data
    #    y_pred = self(x, training=False)
    #    self.compiled_metrics.update_state(, self.pr_bs)
        




















        
