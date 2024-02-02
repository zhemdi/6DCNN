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
import numpy as np


from sympy.physics.quantum.cg import CG


EPS = 1.0e-4
LEAK = 0.05



class conv_params:
    def __init__(self, layers, output_channels):
        """
        Class containing the net architecture
        """
        # Parse the layers list to check for errors
        errors = 0
        for la in layers:
            parse = la.split('_')
            # TODO: improve the errors checking
            if not(parse[0] == 'agg' or parse[0] == 'a6D' or parse[0] == 'ret' or parse[0] == 'full'  or parse[0] == 'afk' or parse[0] == 'a6f'):
                print('ERROR: Anomaly detected while parsing argument :', parse[0],'is not a valid keyword')
                errors += 1
        #if not output_channels[-1] == 3:
        #    print('ERROR: Anomaly detected  :last channel should be 3')
        #    errors += 1
        if not errors == 0:
            raise ValueError(str(errors) + ' error(s) while parsing the argument')
        self.layers = layers
        self.output_channels = output_channels
    def __call__(self):
        print("Layers:")
        for la in self.layers:
            print(la)
        print(self.output_channels)



def _weight_variable(name, shape):
    # return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))
    return tf.Variable(name = name, shape = shape, dtype=tf.float32, initial_value=tf.random.truncated_normal(shape, stddev=0.01))


def _bias_variable(name, shape):
    # return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))
    # return tf.Variable(name = name, shape = shape, dtype=tf.float32, initial_value=0.1*tf.ones(shape, dtype=tf.float32))
    return tf.Variable(name = name, shape = shape, dtype=tf.float32, initial_value=tf.zeros(shape, dtype=tf.float32))

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)

def reduce_sum_in_2_axes(input):
    return tf.reduce_sum(tf.reduce_sum(input,axis = 1), axis = 0)
def reduce_mean_in_2_axes(input):
    return tf.reduce_mean(tf.reduce_mean(input,axis = 1), axis = 0)

class GraphModel:
    def __init__(self,
                layers_param = None,
                num_retypes = 10,
                num_degrees = 10,
                num_radii = 10,
                num_features = 167,
                num_groups = 3,
                normalization = 'group', 
                qa = True):
        

        self.num_retypes = num_retypes
        self.params_conv = layers_param
        self.L = num_degrees - 1
        self.P = num_radii
        self.I = num_features
        self.G = num_groups
        self.input_channels = self.I
        self.agg_normalization = normalization
        self.tensors = []
        self.coefs_N = []
        self.QA = qa


    
    
    
    def predict(self, nodes, groups_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, adjacency_matrices):
        self.precompute_tensors_for_conv6D()
        
        
        for l in range(self.L+1):
            with tf.name_scope("input" + str(l)):
                tf.summary.histogram("input" + str(l) + "_real", nodes[2*l] )
                tf.summary.histogram("input" + str(l) + "_imag", nodes[2*l+1] )


        nodes = self.retyper_block(nodes, internal_layer = False,  output_channel = self.num_retypes, name = 'retyper')

        for l in range(self.L+1):
            with tf.name_scope("retyped" + str(l)):
                tf.summary.histogram("retyped" + str(l) + "_real", nodes[2*l] )
                tf.summary.histogram("retyped" + str(l) + "_imag", nodes[2*l+1] )
        
        nodes = self.functional_block(nodes, groups_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = 'fun_part')
        
        nodes = self.function_to_value(nodes, name = 'embedding')
        with tf.name_scope("vectorized" ):
            tf.summary.histogram("vectorized", nodes)

        fn    = nodes
        
        nodes = self.vector_block(nodes, adjacency_matrices, name = 'vector_part')
        with tf.name_scope("before_last_layer" ):
            tf.summary.histogram("before_last_layer", nodes)

        if self.QA:
            bias_QA = tf.Variable(0.5, name = "bias_QA")
            with tf.name_scope("bias_QA" ):
                tf.summary.scalar("bias_QA", bias_QA)
            dev_QA = tf.Variable(1.0, name = "dev_QA")
            with tf.name_scope("dev_QA" ):
                tf.summary.scalar("dev_QA", dev_QA)
            
            nodes = dev_QA*self.classical_graph_convolution(nodes, adjacency_matrices, output_channel = 1, activation = 'tanh', name = 'final_graph_conv' )+bias_QA 
        else:
            nodes = self.classical_graph_convolution(nodes, adjacency_matrices, output_channel = 3, activation = 'tanh', name = 'final_graph_conv' )
        with tf.name_scope("output" ):
            tf.summary.histogram("output", nodes)
        num_vars = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in tf.trainable_variables()])

        nodes = tf.multiply(nodes, 1, name = 'output_nodes')

        return nodes, fn, num_vars

    def precompute_tensors_for_conv6D(self):
        
        for l in range(self.L+1):
            for l1 in range(self.L+1):
                for l2 in range(np.abs(l -l1),min(self.L,l+l1)+1):
                    self.coefs_N.append(8*np.pi**2/(2*l1+1)*np.sqrt(((2*l+1)*(2*l1+1))/(4*np.pi*(2*l2+1)))*float(CG(l,0,l1,0,l2,0).doit()))
                    T1 = np.zeros((1,l+1, l1+1, l2+1,1,1))
                    T2 = np.zeros((1,l+1, l1+1, l2+1,1,1))
                    T3 = np.zeros((1,l+1, l1+1, l2+1,1,1))
                    for k in range(0, l+1):
                        for k1 in range(0, l1+1):
                            
                            
                            if abs(k+k1) < l2+1:
                                
                                T1[0,k, k1, k+k1,0,0] = float(CG(l,k,l1,k1,l2,k+k1).doit())*(-1)**(k1)
                            if k1 > 0:
                               
                                if abs(k-k1) < l2+1:
                                    if k -k1 >= 0:
                                       
                                        T2[0,k, k1, k-k1,0,0] = float(CG(l,k,l1,-k1,l2,k-k1).doit())
                                    else:
                                       
                                        T3[0,k, k1, k1-k,0,0] = (-1)**(k1-k)*float(CG(l,k,l1,-k1,l2,k-k1).doit())
                    T1_tensor = tf.convert_to_tensor(T1, dtype = tf.float32)
                    T2_tensor = tf.convert_to_tensor(T2, dtype = tf.float32)
                    T3_tensor = tf.convert_to_tensor(T3, dtype = tf.float32)
                    self.tensors.append(T1_tensor)
                    self.tensors.append(T2_tensor)
                    self.tensors.append(T3_tensor)
                    # print(l,l1,l2, len(self.tensors)//3)
        return 

                    




    
    def convolution(self, input,  name = 'convolution', with_bias = False):
        output = []
        for l in range((self.L + 1)):
            weights_degree_real = _weight_variable("weights_" + name + "_" + str(l) + "_real", [1, l+1, self.P, 1])
            weights_degree_imag = _weight_variable("weights_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, 1])
            
            temp_term = np.ones((1, l+1, self.P, 1))
            temp_term[:,0,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            if with_bias:
                bias_degree_real = _bias_variable("biases_" + name + "_" + str(l) + "_real", [1, l+1, self.P, 1])
                bias_degree_imag = _bias_variable("biases_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, 1])
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_real", bias_degree_real )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_imag", bias_degree_imag )
                # temp_term = np.ones((1, l+1, self.P, 1))
                # temp_term[:,0,:,:] = 0
                # temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
                bias_degree_imag = bias_degree_imag*temp_term
                output.append(weights_degree_real*input[2*l] - weights_degree_imag*input[2*l+1]+bias_degree_real)
                output.append(weights_degree_imag*input[2*l] + weights_degree_real*input[2*l+1]+bias_degree_imag)
            else:
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                output.append(weights_degree_real*input[2*l] - weights_degree_imag*input[2*l+1])
                output.append(weights_degree_imag*input[2*l] + weights_degree_real*input[2*l+1])

        return output


    def full_kernel_convolution(self, input, output_channels=10, name = 'full_kernel_convolution', with_bias = False):
        input_channels = input[0].get_shape().as_list()[-1]
        output = []
        for l in range(self.L + 1):
            weights_degree_real = _weight_variable("weights_" + name + "_" + str(l) + "_real", [1, l+1, self.P, input_channels, output_channels])
            weights_degree_imag = _weight_variable("weights_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, input_channels, output_channels])
            
            temp_term = np.ones((1, l+1, self.P, input_channels, output_channels))
            temp_term[:,0,:,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            if with_bias:
                bias_degree_real = _bias_variable("biases_" + name + "_" + str(l) + "_real", [1, l+1, self.P, output_channels])
                bias_degree_imag = _bias_variable("biases_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, output_channels])
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_real", bias_degree_real )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_imag", bias_degree_imag )
                temp_term = np.ones((1, l+1, self.P, output_channels))
                temp_term[:,0,:,:] = 0
                temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
                bias_degree_imag = bias_degree_imag*temp_term
                reshaped_real = tf.reshape(input[2*l], [-1, l+1, self.P, input_channels, 1])
                reshaped_imag = tf.reshape(input[2*l+1], [-1, l+1, self.P, input_channels, 1])
                #output.append(tf.reshape(tf.matmul( tf.reshape(input[l], [-1, input_channels]),tf.reshape(weights_degree, [-1,  input_channels, output_channels])), [-1, 2*l+1, self.P, output_channels])+bias_degree)
                output.append(tf.reduce_sum(weights_degree_real * reshaped_real - weights_degree_imag * reshaped_imag , axis = 3) + bias_degree_real)
                output.append(tf.reduce_sum(weights_degree_real * reshaped_imag + weights_degree_imag * reshaped_real , axis = 3) + bias_degree_imag)
            else:
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                reshaped_real = tf.reshape(input[2*l], [-1, l+1, self.P, input_channels, 1])
                reshaped_imag = tf.reshape(input[2*l+1], [-1, l+1, self.P, input_channels, 1])
                #output.append(tf.reshape(tf.matmul( tf.reshape(input[l], [-1, input_channels]),tf.reshape(weights_degree, [-1,  input_channels, output_channels])), [-1, 2*l+1, self.P, output_channels])+bias_degree)
                output.append(tf.reduce_sum(weights_degree_real * reshaped_real - weights_degree_imag * reshaped_imag , axis = 3) )
                output.append(tf.reduce_sum(weights_degree_real * reshaped_imag + weights_degree_imag * reshaped_real , axis = 3) )
        self.input_channels = output_channels

        return output



    def convolution_in_6D(self, input,  name = 'convolution_6D', with_bias = False):
        weights_list = []
        if with_bias:
            biases_list = []
        input_channels = input[0].get_shape().as_list()[-1]
        
        output = []

        for l in range(self.L+1):
            

            weights_degree_real = _weight_variable("weights_" + name + "_" + str(l) + "_real", [1, 1, l+1, self.P, 1])
            weights_degree_imag = _weight_variable("weights_" + name + "_" + str(l) + "_imag", [1, 1, l+1, self.P, 1])
            
            temp_term = np.ones((1, 1, l+1, self.P, 1))
            temp_term[:,:,0,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            weights_list.append(weights_degree_real)
            weights_list.append(weights_degree_imag)

            if with_bias:
                bias_degree_real = _bias_variable("biases_" + name + "_" + str(l) + "_real", [1, l+1, self.P, 1])
                bias_degree_imag = _bias_variable("biases_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, 1])
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_real", bias_degree_real )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_imag", bias_degree_imag )
                temp_term_bias = np.ones((1, l+1, self.P,1))
                temp_term_bias[:,0,:,:] = 0
                temp_term_bias = tf.convert_to_tensor(temp_term_bias, dtype = tf.float32)
                bias_degree_imag = bias_degree_imag*temp_term_bias
                biases_list.append(bias_degree_real)
                biases_list.append(bias_degree_imag)
            else:
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
        
        itr = 0
        for l in range(self.L+1):
            output_i_real = 0.0
            output_i_imag = 0.0
            for l1 in range(self.L+1):
                for l2 in range(np.abs(l -l1),min(self.L,l+l1)+1):
                    coef = self.coefs_N[itr]
                    interm_term_f_real = tf.reduce_sum((self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels]), axis = 3)
                    interm_term_s_real = tf.reduce_sum((self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels]), axis = 3)
                    interm_term_f_imag = tf.reduce_sum((self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels]), axis = 3)
                    interm_term_s_imag = tf.reduce_sum((self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels]), axis = 3)
                    output_i_real += coef*tf.reduce_sum(interm_term_f_real*weights_list[2*l1] - interm_term_s_real*weights_list[2*l1+1] , axis = 2)
                    output_i_imag += coef*tf.reduce_sum(interm_term_f_imag*weights_list[2*l1+1] + interm_term_s_imag*weights_list[2*l1] , axis = 2)
                    itr +=1

            if with_bias:
                output.append(output_i_real + biases_list[2*l])
                output.append(output_i_imag + biases_list[2*l+1])
            else:
                output.append(output_i_real)
                output.append(output_i_imag)
        if with_bias:
            return output, biases_list
        return output




    def full_convolution_in_6D(self, input, output_channels=10,  name = 'full_convolution_6D', with_bias = False):
        weights_list = []
        if with_bias:
            biases_list = []
        input_channels = input[0].get_shape().as_list()[-1]
        
        output = []

        for l in range(self.L+1):
            

            weights_degree_real = _weight_variable("weights_" + name + "_" + str(l) + "_real", [1, 1, l+1, self.P, input_channels, output_channels])
            weights_degree_imag = _weight_variable("weights_" + name + "_" + str(l) + "_imag", [1, 1, l+1, self.P, input_channels, output_channels])
            # with tf.name_scope(name):
            #     tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
            #     tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
            temp_term = np.ones((1, 1, l+1, self.P, 1,1))
            temp_term[:,:,0,:,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            weights_list.append(weights_degree_real)
            weights_list.append(weights_degree_imag)

            if with_bias:
                bias_degree_real = _bias_variable("biases_" + name + "_" + str(l) + "_real", [1, l+1, self.P, output_channels])
                bias_degree_imag = _bias_variable("biases_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, output_channels])
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_real", bias_degree_real )
                    tf.summary.histogram("biases_" + name + "_" + str(l) + "_imag", bias_degree_imag)
                temp_term_bias = np.ones((1, l+1, self.P,1))
                temp_term_bias[:,0,:,:] = 0
                temp_term_bias = tf.convert_to_tensor(temp_term_bias, dtype = tf.float32)
                bias_degree_imag = bias_degree_imag*temp_term_bias
                biases_list.append(bias_degree_real)
                biases_list.append(bias_degree_imag)
            else:
                with tf.name_scope(name + "_" + str(l)):
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                    tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
        
        itr = 0
        for l in range(self.L+1):
            output_i_real = 0.0
            output_i_imag = 0.0
            for l1 in range(self.L+1):
                for l2 in range(np.abs(l -l1),min(self.L,l+l1)+1):
                    coef = self.coefs_N[itr]
                    # print(l,l1,l2, itr+1)
                    # interm_term_f_real = tf.reduce_sum(tf.reshape(self.tensors[3*itr] +self.tensors[3*itr+1]+self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)
                    # interm_term_s_real = tf.reduce_sum(tf.reshape(self.tensors[3*itr] -self.tensors[3*itr+1]+self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)
                    # interm_term_f_imag = tf.reduce_sum(tf.reshape(self.tensors[3*itr] -self.tensors[3*itr+1]-self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)
                    # interm_term_s_imag = tf.reduce_sum(tf.reshape(self.tensors[3*itr] +self.tensors[3*itr+1]-self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)
                    # output_i_real += coef*tf.reduce_sum(tf.reduce_sum(interm_term_f_real*weights_list[2*l1] - interm_term_s_real*weights_list[2*l1+1] , axis = 2), axis = 3)
                    # output_i_imag += coef*tf.reduce_sum(tf.reduce_sum(interm_term_f_imag*weights_list[2*l1+1] + interm_term_s_imag*weights_list[2*l1] , axis = 2), axis = 3)
                    
                    output_i_real += coef*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reshape(self.tensors[3*itr] + self.tensors[3*itr+1] + self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)*weights_list[2*l1] - tf.reduce_sum(tf.reshape(self.tensors[3*itr] - self.tensors[3*itr+1] + self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)*weights_list[2*l1+1] , axis = 2), axis = 3)
                    output_i_imag += coef*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reshape(self.tensors[3*itr] - self.tensors[3*itr+1] - self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)*weights_list[2*l1+1] + tf.reduce_sum(tf.reshape(self.tensors[3*itr] + self.tensors[3*itr+1] - self.tensors[3*itr+2], [1,l+1,l1+1,l2+1, 1,1,1])*tf.reshape(input[2*l2+1], [-1, 1,1,l2+1, self.P, input_channels,1]), axis = 3)*weights_list[2*l1] , axis = 2), axis = 3)
                    
                    
                    itr+=1
            
            if with_bias:
                output.append(output_i_real + biases_list[2*l])
                output.append(output_i_imag + biases_list[2*l+1])
            else:
                output.append(output_i_real)
                output.append(output_i_imag)
        if with_bias:
            return output, biases_list
        return output

        

                    
                    






        

            
            



    
    
    
    def retyper(self, input, with_bias = False, output_channels = 10, name = 'retyper'):
        input_channels = input[0].get_shape().as_list()[-1]
        retyper_real = _weight_variable("retyper_" + name + "_real", [input_channels, output_channels])
        # retyper_imag = _weight_variable("retyper_" + name + "_imag", [input_channels, output_channels])
        retyper_imag = 0.0
        
        if with_bias:
            bias_real =  _bias_variable("biases_" + name + "_real" , [output_channels])
            bias_imag =  _bias_variable("biases_" + name + "_imag" , [output_channels])
            with tf.name_scope(name):
                tf.summary.histogram("retyper_" + name + "_real", retyper_real )
                # tf.summary.histogram("retyper_" + name + "_imag", retyper_imag )
                tf.summary.histogram("biases_" + name + "_real", bias_real )
                tf.summary.histogram("biases_" + name + "_imag", bias_imag )
        else:
            with tf.name_scope(name):
                tf.summary.histogram("retyper_" + name + "_real", retyper_real )
                # tf.summary.histogram("retyper_" + name + "_imag", retyper_imag )
        self.input_channels = output_channels
        
        output = []
        for l in range(self.L+1):
            if with_bias:
                # output.append(tf.matmul(input[2*l], retyper_real) - tf.matmul(input[2*l+1], retyper_imag) + bias_real)
                # output.append(tf.matmul(input[2*l], retyper_imag) + tf.matmul(input[2*l+1], retyper_real) + bias_imag)
                output.append(tf.matmul(input[2*l], retyper_real)  + bias_real)
                output.append(tf.matmul(input[2*l+1], retyper_real)+ bias_imag)
            else:
                # output.append(tf.matmul(input[2*l], retyper_real) - tf.matmul(input[2*l+1], retyper_imag))
                # output.append(tf.matmul(input[2*l], retyper_imag) + tf.matmul(input[2*l+1], retyper_real))
                output.append(tf.matmul(input[2*l], retyper_real))
                output.append(tf.matmul(input[2*l+1], retyper_real))


        return output


    def filter_activation(self,signal, processed_signal, name = 'filter_activation'):
        input_channels = signal[0].get_shape().as_list()[-1]
        result_node = 0.0
        processed_result_node = 0.0
        for l in range(self.L + 1):
            result_node += tf.reduce_mean(tf.reduce_mean(tf.square(signal[2*l]), axis = 2), axis = 1)
            result_node += tf.reduce_mean(tf.reduce_mean(tf.square(signal[2*l+1]), axis = 2), axis = 1)
            processed_result_node += tf.reduce_mean(tf.reduce_mean(tf.square(processed_signal[2*l]), axis = 2), axis = 1)
            processed_result_node += tf.reduce_mean(tf.reduce_mean(tf.square(processed_signal[2*l+1]), axis = 2), axis = 1)
        condition = tf.less(result_node, processed_result_node)
        condition = tf.cast(condition, tf.float32)
        condition = tf.reshape(condition, [-1, 1, 1, input_channels])
        output = []
        for l in range(self.L+1):
            output.append(signal[2*l] + (LEAK + (1 - LEAK)*condition)*(processed_signal[2*l] - signal[2*l]))
            output.append(signal[2*l+1] + (LEAK + (1 - LEAK)*condition)*(processed_signal[2*l+1] - signal[2*l+1]))
        


        return output
    
    def function_activation(self,processed_signal, name = 'function_activation'):
        input_channels = processed_signal[0].get_shape().as_list()[-1]
        w = np.zeros((1,1, self.P, 1))
        w[:,0,0,:] = 1.0
        w_tensor = tf.convert_to_tensor(w, dtype = tf.float32, name = name + 'extract_first_coef')
        first_coef = tf.reshape(tf.reduce_sum(tf.multiply(w_tensor, processed_signal[0]), axis = 2),[-1, 1, 1, input_channels])
        condition = tf.less(tf.cast(0.0,tf.float32), first_coef)
        condition = tf.cast(condition, tf.float32)
        output = []
        for l in range(self.L+1):
            output.append((LEAK + (1 - LEAK)*condition)*processed_signal[2*l])
            output.append((LEAK + (1 - LEAK)*condition)*processed_signal[2*l+1])
        


        return output


    def function_activation2(self,processed_signal, biases, name = 'function_activation2'):
        input_channels = processed_signal[0].get_shape().as_list()[-1]
        processed_result_node = 0.0
        
        for l in range(self.L + 1):
            processed_result_node += tf.reduce_sum(tf.reduce_sum((processed_signal[2*l]**2 - biases[2*l]**2), axis = 2), axis = 1)
            processed_result_node += tf.reduce_sum(tf.reduce_sum((processed_signal[2*l+1]**2 - biases[2*l+1]**2), axis = 2), axis = 1)
            if l > 0:
                processed_result_node += tf.reduce_sum(tf.reduce_sum((processed_signal[2*l]**2 - biases[2*l]**2), axis = 2), axis = 1)
                processed_result_node += tf.reduce_sum(tf.reduce_sum((processed_signal[2*l+1]**2 - biases[2*l+1]**2), axis = 2), axis = 1)
        condition = tf.less_equal(0.0, processed_result_node)
        condition = tf.cast(condition, tf.float32)
        condition = tf.reshape(condition, [-1, 1, 1, input_channels])
        processed_result_node_reshaped = tf.reshape(processed_result_node, [-1,1,1, input_channels])
        output = []
        for l in range(self.L+1):
            output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l] ) + (1 - condition)*(1 - LEAK)/(processed_signal[2*l]**2 - processed_result_node_reshaped)*processed_signal[2*l]**3)
            output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l+1] ) + (1 - condition)*(1 - LEAK)/(processed_signal[2*l+1]**2 - processed_result_node_reshaped)*processed_signal[2*l+1]**3)
            # output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l] ))
            # output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l+1] ))
        return output


    def function_activation3(self,processed_signal, biases, name = 'function_activation3'):
        input_channels = processed_signal[0].get_shape().as_list()[-1]
        processed_result_node = 0.0
        
        for l in range(self.L + 1):
            processed_result_node += tf.reduce_sum(tf.reduce_mean(tf.square(processed_signal[2*l]- biases[2*l]), axis = 2), axis = 1)
            processed_result_node += tf.reduce_sum(tf.reduce_mean(tf.square(processed_signal[2*l+1]- biases[2*l+1]), axis = 2), axis = 1)
            if l > 0:
                processed_result_node += tf.reduce_sum(tf.reduce_mean(tf.square(processed_signal[2*l] - biases[2*l]), axis = 2), axis = 1)
                processed_result_node += tf.reduce_sum(tf.reduce_mean(tf.square(processed_signal[2*l+1] - biases[2*l+1]), axis = 2), axis = 1)
        # condition = 1.0 - 0.25*processed_result_node
        condition = 0.25*processed_result_node/((self.L+1)**2)
        with tf.name_scope(name + "_condition" ):
            tf.summary.histogram(name + "_condition", condition)
        condition = tf.reshape(condition, [-1, 1, 1, input_channels])
        
        output = []
        for l in range(self.L+1):
            # output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l] ) + (1 - condition)*(1 - LEAK)*(1.0/((processed_signal[2*l]-biases[2*l])**2 +2 - processed_result_node_reshaped)*(processed_signal[2*l]-biases[2*l])**3 + biases[2*l]))
            # output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l+1] ) + (1 - condition)*(1 - LEAK)*(1.0/((processed_signal[2*l+1] - biases[2*l+1])**2 + 2 - processed_result_node_reshaped)*(processed_signal[2*l+1] - biases[2*l+1])**3 + biases[2*l+1]))
            output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l] ))
            output.append((LEAK + (1 - LEAK)*condition)*(processed_signal[2*l+1] ))
            
        return output


    def normalization(self,signal, gamma = 1, beta = 1.0e-5, name = 'normalization'):
        input_channels = signal[0].get_shape().as_list()[-1]
        #print(signal[0].get_shape().as_list())
        # gamma = tf.Variable(1.0, trainable=True, name=name+"_gamma")
        # beta = tf.Variable(0.01, trainable=True, name=name+"_beta")
        # with tf.name_scope(name):
        #     tf.summary.scalar('gamma_' + name, gamma)
        #     tf.summary.scalar('beta_' + name, beta)
        sum_value_real = 0.0
        sum_value_imag = 0.0
        sum_sq_value = 0.0
        for l in range(self.L+1):
            sum_value_real += tf.reshape(tf.reduce_sum(tf.reduce_mean(signal[2*l], axis = 2), axis = 1),[-1,1,1,input_channels])
            sum_value_imag += tf.reshape(tf.reduce_sum(tf.reduce_mean(signal[2*l+1], axis = 2), axis = 1),[-1,1,1,input_channels])
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l]), axis = 2), axis = 1),[-1,1,1,input_channels])
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l+1]), axis = 2), axis = 1),[-1,1,1,input_channels])
            
            if l > 0:
                MoK = np.array([(-1)**k for k in range(l+1)])
                MoK_tensor = tf.reshape(tf.convert_to_tensor(MoK, dtype = tf.float32), [1,l+1,1,1])
                sum_value_real +=  tf.reshape(tf.reduce_sum(tf.reduce_mean(MoK_tensor*signal[2*l], axis = 2), axis = 1),[-1,1,1,input_channels])
                sum_value_imag += -tf.reshape(tf.reduce_sum(tf.reduce_mean(MoK_tensor*signal[2*l+1], axis = 2), axis = 1),[-1,1,1,input_channels])
                sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l]), axis = 2), axis = 1),[-1,1,1,input_channels])
                sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l+1]), axis = 2), axis = 1),[-1,1,1,input_channels])
        mean_real = sum_value_real/((self.L+1)**2)
        mean_imag = sum_value_imag/((self.L+1)**2)
        var = sum_sq_value/((self.L+1)**2) - tf.square(sum_value_real/((self.L+1)**2)) - tf.square(sum_value_imag/((self.L+1)**2))
        output = []
        for l in range(self.L+1):
            output.append(gamma*(signal[2*l] - mean_real)/tf.sqrt(var+EPS) + beta)
            output.append(gamma*(signal[2*l+1] - mean_imag)/tf.sqrt(var+EPS) + beta)
            #output.append(signal[l]/tf.sqrt(sum_sq_value+EPS))
        return output



    def normalization2(self,signal, gamma = 1, beta = 1.0e-5, name = 'normalization'):
        input_channels = signal[0].get_shape().as_list()[-1]
        
        w = np.zeros((1,1, self.P, 1))
        w[:,0,0,:] = 1.0
        w_tensor = tf.convert_to_tensor(w, dtype = tf.float32, name = name + 'extract_first_coef')
        first_coef = tf.reshape(tf.reduce_sum(tf.multiply(w_tensor, signal[0]), axis = 2),[-1, 1, 1, input_channels])
        first_coef_im = tf.reshape(tf.reduce_sum(tf.multiply(w_tensor, signal[1]), axis = 2),[-1, 1, 1, input_channels])
        sum_sq_value  = tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[0] -first_coef   ), axis = 2), axis = 1),[-1,1,1,input_channels])
        sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[1] -first_coef_im), axis = 2), axis = 1),[-1,1,1,input_channels])
        for l in range(1, self.L+1):
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l] -first_coef), axis = 2), axis = 1),[-1,1,1,input_channels])
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(signal[2*l+1]), axis = 2), axis = 1),[-1,1,1,input_channels])
            MoK = np.array([(-1)**k for k in range(l+1)])
            MoK_tensor = tf.reshape(tf.convert_to_tensor(MoK, dtype = tf.float32), [1,l+1,1,1])
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(MoK_tensor*signal[2*l  ] - first_coef   ), axis = 2), axis = 1),[-1,1,1,input_channels])
            sum_sq_value += tf.reshape(tf.reduce_sum(tf.reduce_mean(tf.square(MoK_tensor*signal[2*l+1] - first_coef_im), axis = 2), axis = 1),[-1,1,1,input_channels])
        var = sum_sq_value/((self.L+1)**2)
        output = []
        for l in range(self.L+1):
            output.append(gamma*(signal[2*l  ] - first_coef   )/tf.sqrt(var+EPS) + beta)
            output.append(gamma*(signal[2*l+1] - first_coef_im)/tf.sqrt(var+EPS) + beta)
            #output.append(signal[l]/tf.sqrt(sum_sq_value+EPS))
        return output


    def aggregation_block(self, input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = 'agg_block'):

        input_channels = input[0].get_shape().as_list()[-1]
        output = []
        for g in range(self.G):
            N_to_E = groups_of_edges[2*g]
            E_to_N = groups_of_edges[2*g+1]
            first_rotation_and_translation = []
            for l1 in range(self.L +1):
                print(l1)
                reshaped_real  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                reshaped_imag  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                fsm_l_pos_real = tf.transpose(first_slater_matrices[4*l1  ], [0,1,2])
                fsm_l_pos_imag = tf.transpose(first_slater_matrices[4*l1+1], [0,1,2])
                fsm_l_neg_real = tf.transpose(first_slater_matrices[4*l1+2], [0,1,2])
                fsm_l_neg_imag = tf.transpose(first_slater_matrices[4*l1+3], [0,1,2])
                rotated_real   = tf.matmul(fsm_l_pos_real, reshaped_real) + tf.matmul(fsm_l_pos_imag, reshaped_imag)
                rotated_imag   = tf.matmul(fsm_l_pos_real, reshaped_imag) - tf.matmul(fsm_l_pos_imag, reshaped_real)
                rotated_real  += tf.matmul(fsm_l_neg_real, reshaped_real) - tf.matmul(fsm_l_neg_imag, reshaped_imag)
                rotated_imag  += 0-tf.matmul(fsm_l_neg_real, reshaped_imag) - tf.matmul(fsm_l_neg_imag, reshaped_real)
                # rotated_real = tf.matmul(first_slater_matrices[4*l1], reshaped_real) - tf.matmul(first_slater_matrices[4*l1+1], reshaped_imag)
                # rotated_imag = tf.matmul(first_slater_matrices[4*l1+1], reshaped_real) + tf.matmul(first_slater_matrices[4*l1], reshaped_imag)
                rotated_reshaped_real = tf.reshape(rotated_real, [-1, (l1+1), self.P, input_channels])
                rotated_reshaped_real = tf.transpose(rotated_reshaped_real, [0,2,1,3])
                rotated_reshaped_real = tf.reshape(rotated_reshaped_real, [-1, (l1+1), input_channels])
                rotated_reshaped_imag = tf.reshape(rotated_imag, [-1, (l1+1), self.P, input_channels])
                rotated_reshaped_imag = tf.transpose(rotated_reshaped_imag, [0,2,1,3])
                rotated_reshaped_imag = tf.reshape(rotated_reshaped_imag, [-1, (l1+1), input_channels])


                # reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(rotated_real, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                # reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(rotated_imag, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                for l2 in range(self.L+1):
                    bessel_matrix_real = bessel_matrices[2*(l1*(self.L+1)+l2)]
                    bessel_matrix_imag = bessel_matrices[2*(l1*(self.L+1)+l2)+1]
                    bessel_matrix_reshaped_real = tf.reshape(bessel_matrix_real, [-1, (l2+1), (l1+1)])
                    bessel_matrix_reshaped_imag = tf.reshape(bessel_matrix_imag, [-1, (l2+1), (l1+1)])
                    term_dd_real = tf.matmul(bessel_matrix_reshaped_real, rotated_reshaped_real) - tf.matmul(bessel_matrix_reshaped_imag, rotated_reshaped_imag)
                    term_dd_imag = tf.matmul(bessel_matrix_reshaped_real, rotated_reshaped_imag) + tf.matmul(bessel_matrix_reshaped_imag, rotated_reshaped_real)
                    # term_dd_real = tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)], rotated_real) - tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)+1], rotated_imag)
                    # term_dd_imag = tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)], rotated_imag) + tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)+1], rotated_real)

                    term_dd_real = tf.reshape(tf.transpose(tf.reshape(term_dd_real, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                    term_dd_imag = tf.reshape(tf.transpose(tf.reshape(term_dd_imag, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                    
                    
                    
                    if len(first_rotation_and_translation) < 2*(l2+1):
                        first_rotation_and_translation.append(term_dd_real)
                        first_rotation_and_translation.append(term_dd_imag)
                    else:
                        first_rotation_and_translation[2*l2] += term_dd_real
                        first_rotation_and_translation[2*l2+1] += term_dd_imag
            
            shifted = []
            for l in range(self.L+1):
                second_rotation_real  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l]) + tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1])
                second_rotation_imag  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l])
                second_rotation_real += tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1])
                second_rotation_imag += 0-tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l])
                if self.agg_normalization == 'nodes':
                    term_real = (input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    term_imag = (input[2*l+1] +tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    # term_real = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    # term_imag = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    
                else:
                    term_real = input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    term_imag = input[2*l+1] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_real = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_imag = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                
                shifted.append( term_real)
                shifted.append( term_imag)
                    
            convoluted = self.convolution(shifted, name = name + '_' + str(g) + '_convolution')
            #convoluted = self.full_kernel_convolution(shifted, output_channels = input_channels, name = name + '_' + str(g) + '_convolution')
            if self.agg_normalization == 'nodes':
                normalized = convoluted
            else:
                normalized = self.normalization(convoluted, name = name + '_' + str(g) + '_normalization')
            output_group = self.filter_activation(input, normalized, name = name + '_' + str(g) + '_filter_activation')
            for l in range(self.L+1):
                output_group_degree_real = output_group[2*l]/self.G
                output_group_degree_imag = output_group[2*l+1]/self.G
                if len(output) < 2*(self.L+1):
                    output.append(output_group_degree_real)
                    output.append(output_group_degree_imag)
                    
                else:
                    output[2*l] += output_group_degree_real
                    output[2*l+1] += output_group_degree_imag


                
            
        return output


    

    def aggregation_nc_block(self, input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = 'agg_nc_block'):

        input_channels = input[0].get_shape().as_list()[-1]
        output = []
        for g in range(self.G):
            N_to_E = groups_of_edges[2*g]
            E_to_N = groups_of_edges[2*g+1]
            first_rotation_and_translation = []
            for l1 in range(self.L +1):
                print(l1)
                reshaped_real  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                reshaped_imag  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                fsm_l_pos_real = tf.transpose(first_slater_matrices[4*l1  ], [0,1,2])
                fsm_l_pos_imag = tf.transpose(first_slater_matrices[4*l1+1], [0,1,2])
                fsm_l_neg_real = tf.transpose(first_slater_matrices[4*l1+2], [0,1,2])
                fsm_l_neg_imag = tf.transpose(first_slater_matrices[4*l1+3], [0,1,2])
                rotated_real   = tf.matmul(fsm_l_pos_real, reshaped_real) + tf.matmul(fsm_l_pos_imag, reshaped_imag)
                rotated_imag   = tf.matmul(fsm_l_pos_real, reshaped_imag) - tf.matmul(fsm_l_pos_imag, reshaped_real)
                rotated_real  += tf.matmul(fsm_l_neg_real, reshaped_real) - tf.matmul(fsm_l_neg_imag, reshaped_imag)
                rotated_imag  += 0-tf.matmul(fsm_l_neg_real, reshaped_imag) - tf.matmul(fsm_l_neg_imag, reshaped_real)
                # rotated_real = tf.matmul(first_slater_matrices[4*l1], reshaped_real) - tf.matmul(first_slater_matrices[4*l1+1], reshaped_imag)
                # rotated_imag = tf.matmul(first_slater_matrices[4*l1+1], reshaped_real) + tf.matmul(first_slater_matrices[4*l1], reshaped_imag)
                rotated_reshaped_real = tf.reshape(rotated_real, [-1, (l1+1), self.P, input_channels])
                rotated_reshaped_real = tf.transpose(rotated_reshaped_real, [0,2,1,3])
                rotated_reshaped_real = tf.reshape(rotated_reshaped_real, [-1, (l1+1), input_channels])
                rotated_reshaped_imag = tf.reshape(rotated_imag, [-1, (l1+1), self.P, input_channels])
                rotated_reshaped_imag = tf.transpose(rotated_reshaped_imag, [0,2,1,3])
                rotated_reshaped_imag = tf.reshape(rotated_reshaped_imag, [-1, (l1+1), input_channels])


                # reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(rotated_real, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                # reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(rotated_imag, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                for l2 in range(self.L+1):
                    bessel_matrix_real = bessel_matrices[2*(l1*(self.L+1)+l2)]
                    bessel_matrix_imag = bessel_matrices[2*(l1*(self.L+1)+l2)+1]
                    bessel_matrix_reshaped_real = tf.reshape(bessel_matrix_real, [-1, (l2+1), (l1+1)])
                    bessel_matrix_reshaped_imag = tf.reshape(bessel_matrix_imag, [-1, (l2+1), (l1+1)])
                    term_dd_real = tf.matmul(bessel_matrix_reshaped_real, rotated_reshaped_real) - tf.matmul(bessel_matrix_reshaped_imag, rotated_reshaped_imag)
                    term_dd_imag = tf.matmul(bessel_matrix_reshaped_real, rotated_reshaped_imag) + tf.matmul(bessel_matrix_reshaped_imag, rotated_reshaped_real)
                    # term_dd_real = tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)], rotated_real) - tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)+1], rotated_imag)
                    # term_dd_imag = tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)], rotated_imag) + tf.matmul(bessel_matrices[2*(l1*(self.L+1)+l2)+1], rotated_real)

                    term_dd_real = tf.reshape(tf.transpose(tf.reshape(term_dd_real, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                    term_dd_imag = tf.reshape(tf.transpose(tf.reshape(term_dd_imag, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                    
                    
                    
                    if len(first_rotation_and_translation) < 2*(l2+1):
                        first_rotation_and_translation.append(term_dd_real)
                        first_rotation_and_translation.append(term_dd_imag)
                    else:
                        first_rotation_and_translation[2*l2] += term_dd_real
                        first_rotation_and_translation[2*l2+1] += term_dd_imag
            
            shifted = []
            for l in range(self.L+1):
                second_rotation_real  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l]) + tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1])
                second_rotation_imag  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l])
                second_rotation_real += tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1])
                second_rotation_imag += 0-tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l])
                if self.agg_normalization == 'nodes':
                    term_real = (input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    term_imag = (input[2*l+1] +tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    # term_real = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    # term_imag = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    
                else:
                    term_real = input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    term_imag = input[2*l+1] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_real = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_imag = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                
                shifted.append( term_real)
                shifted.append( term_imag)
                    
            convoluted, biases = self.convolution_in_6D(shifted, name = name + '_' + str(g) + '_convolution', with_bias=True)
            #convoluted = self.full_kernel_convolution(shifted, output_channels = input_channels, name = name + '_' + str(g) + '_convolution')
            # if self.agg_normalization == 'nodes':
            #     normalized = convoluted
            # else:
            #     normalized = self.normalization(convoluted, name = name + '_' + str(g) + '_normalization')
            # output_group = self.filter_activation(input, normalized, name = name + '_' + str(g) + '_filter_activation')
            normalized_tensor = self.normalization(convoluted, name = name + '_' + str(g) + '_normalization')
            normalized_biases = self.normalization(biases, name = name + '_' + str(g) + '_normalization_bias')
            output_group = self.function_activation2(normalized_tensor, normalized_biases, name = name + '_' + str(g) + '_function_activation2')

            for l in range(self.L+1):
                output_group_degree_real = output_group[2*l]/self.G
                output_group_degree_imag = output_group[2*l+1]/self.G
                if len(output) < 2*(self.L+1):
                    output.append(output_group_degree_real)
                    output.append(output_group_degree_imag)
                    
                else:
                    output[2*l] += output_group_degree_real
                    output[2*l+1] += output_group_degree_imag


                
            
        return output



    def full_kernel_aggregation_block(self, input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, output_channel = 10, name = 'fk_agg_block'):
        
        input_channels = input[0].get_shape().as_list()[-1]
        output = []
        for g in range(self.G):
            N_to_E = groups_of_edges[2*g]
            E_to_N = groups_of_edges[2*g+1]
            first_rotation_and_translation = []
            for l1 in range(self.L +1):
                print(l1)
                reshaped_real  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                reshaped_imag  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                fsm_l_pos_real = tf.transpose(first_slater_matrices[4*l1  ], [0,1,2])
                fsm_l_pos_imag = tf.transpose(first_slater_matrices[4*l1+1], [0,1,2])
                fsm_l_neg_real = tf.transpose(first_slater_matrices[4*l1+2], [0,1,2])
                fsm_l_neg_imag = tf.transpose(first_slater_matrices[4*l1+3], [0,1,2])
                rotated_real   = tf.matmul(fsm_l_pos_real, reshaped_real) + tf.matmul(fsm_l_pos_imag, reshaped_imag)
                rotated_imag   = tf.matmul(fsm_l_pos_real, reshaped_imag) - tf.matmul(fsm_l_pos_imag, reshaped_real)
                rotated_real  += tf.matmul(fsm_l_neg_real, reshaped_real) - tf.matmul(fsm_l_neg_imag, reshaped_imag)
                rotated_imag  += 0-tf.matmul(fsm_l_neg_real, reshaped_imag) - tf.matmul(fsm_l_neg_imag, reshaped_real)
                # rotated_real = tf.matmul(first_slater_matrices[2*l1], reshaped_real) - tf.matmul(first_slater_matrices[2*l1+1], reshaped_imag)
                # rotated_imag = tf.matmul(first_slater_matrices[2*l1+1], reshaped_real) + tf.matmul(first_slater_matrices[2*l1], reshaped_imag)
                reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(rotated_real, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(rotated_imag, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                for l2 in range(self.L+1):
                    bm_real = tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)  ], [-1, l2+1, l1+1])
                    bm_imag = tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)+1], [-1, l2+1, l1+1])
                    term_dd_real = tf.matmul(bm_real, reshaped_transposed_real) - tf.matmul(bm_imag, reshaped_transposed_imag)
                    term_dd_imag = tf.matmul(bm_real, reshaped_transposed_imag) + tf.matmul(bm_imag, reshaped_transposed_real)

                    term_dd_real = tf.reshape(tf.transpose(tf.reshape(term_dd_real, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                    term_dd_imag = tf.reshape(tf.transpose(tf.reshape(term_dd_imag, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])


                    if len(first_rotation_and_translation) < 2*(l2+1):
                        first_rotation_and_translation.append(term_dd_real)
                        first_rotation_and_translation.append(term_dd_imag)
                    else:
                        first_rotation_and_translation[2*l2] += term_dd_real
                        first_rotation_and_translation[2*l2+1] += term_dd_imag
            
            shifted = []
            for l in range(self.L+1):
                second_rotation_real  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l]) + tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1])
                second_rotation_imag  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l])
                second_rotation_real += tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1])
                second_rotation_imag += 0-tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l])
                
                # second_rotation_real = tf.matmul(second_slater_matrices[2*l], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[2*l+1], first_rotation_and_translation[2*l+1])
                # second_rotation_imag = tf.matmul(second_slater_matrices[2*l], first_rotation_and_translation[2*l+1]) + tf.matmul(second_slater_matrices[2*l+1], first_rotation_and_translation[2*l])
                if self.agg_normalization == 'nodes':
                    term_real = (input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    term_imag = (input[2*l+1] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1)+1,[-1,1,1,1]))
                    # term_real = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    # term_imag = (tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels]))/(tf.reshape(tf.reduce_sum(E_to_N, axis = 1),[-1,1,1,1]))
                    
                else:
                    term_real = input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    term_imag = input[2*l+1] +tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_real = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_imag = tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                
                shifted.append( term_real)
                shifted.append( term_imag)
                    
            convoluted = self.full_kernel_convolution(shifted, output_channels = output_channel, name = name + '_' + str(g) + '_convolution')
            #convoluted = self.full_kernel_convolution(shifted, output_channels = input_channels, name = name + '_' + str(g) + '_convolution')
            if self.agg_normalization == 'nodes':
                normalized = convoluted
            else:
                normalized = self.normalization(convoluted, name = name + '_' + str(g) + '_normalization')
            output_group = self.function_activation( normalized, name = name + '_' + str(g) + '_function_activation')
            for l in range(self.L+1):
                output_group_degree_real = output_group[2*l]/self.G
                output_group_degree_imag = output_group[2*l+1]/self.G
                if len(output) < 2*(self.L+1):
                    output.append(output_group_degree_real)
                    output.append(output_group_degree_imag)
                    
                else:
                    output[2*l] += output_group_degree_real
                    output[2*l+1] += output_group_degree_imag


                
            
        return output



    def full_kernel_aggregation_nc_block(self, input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, do_aggregation = True, output_channel = 10, name = 'fk_agg_nc_block'):
        
        input_channels = input[0].get_shape().as_list()[-1]
        output = []
        for g in range(self.G):
            if do_aggregation:
                N_to_E = groups_of_edges[2*g]
                E_to_N = groups_of_edges[2*g+1]
                first_rotation_and_translation = []
                for l1 in range(self.L +1):
                    # print(l1)
                    # reshaped_real  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                    # reshaped_imag  = tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])
                    # fsm_l_pos_real = tf.transpose(first_slater_matrices[4*l1  ], [0,1,2])
                    # fsm_l_pos_imag = tf.transpose(first_slater_matrices[4*l1+1], [0,1,2])
                    # fsm_l_neg_real = tf.transpose(first_slater_matrices[4*l1+2], [0,1,2])
                    # fsm_l_neg_imag = tf.transpose(first_slater_matrices[4*l1+3], [0,1,2])
                    # rotated_real   = tf.matmul(fsm_l_pos_real, reshaped_real) + tf.matmul(fsm_l_pos_imag, reshaped_imag)
                    # rotated_imag   = tf.matmul(fsm_l_pos_real, reshaped_imag) - tf.matmul(fsm_l_pos_imag, reshaped_real)
                    # rotated_real  += tf.matmul(fsm_l_neg_real, reshaped_real) - tf.matmul(fsm_l_neg_imag, reshaped_imag)
                    # rotated_imag  += 0-tf.matmul(fsm_l_neg_real, reshaped_imag) - tf.matmul(fsm_l_neg_imag, reshaped_real)
                    # reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(rotated_real, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                    # reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(rotated_imag, [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                    
                    reshaped_transposed_real  = tf.reshape(tf.transpose(tf.reshape(tf.matmul(tf.transpose(first_slater_matrices[4*l1  ], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) + tf.matmul(tf.transpose(first_slater_matrices[4*l1+1], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) + tf.matmul(tf.transpose(first_slater_matrices[4*l1+2], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) - tf.matmul(tf.transpose(first_slater_matrices[4*l1+3], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])), [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                    reshaped_transposed_imag  = tf.reshape(tf.transpose(tf.reshape(tf.matmul(tf.transpose(first_slater_matrices[4*l1  ], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) - tf.matmul(tf.transpose(first_slater_matrices[4*l1+1], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) - tf.matmul(tf.transpose(first_slater_matrices[4*l1+2], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1+1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])) - tf.matmul(tf.transpose(first_slater_matrices[4*l1+3], [0,1,2]), tf.reshape(tf.matmul(N_to_E,tf.reshape(input[2*l1],  [-1, (l1+1)*self.P*input_channels])), [-1, l1+1, self.P*input_channels])), [-1, l1+1, self.P, input_channels]), [0,2,1,3]), [-1, l1+1, input_channels])
                    
                    
                    for l2 in range(self.L+1):
                        # bm_real = tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)  ], [-1, l2+1, l1+1])
                        # bm_imag = tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)+1], [-1, l2+1, l1+1])
                        # term_dd_real = tf.matmul(bm_real, reshaped_transposed_real) - tf.matmul(bm_imag, reshaped_transposed_imag)
                        # term_dd_imag = tf.matmul(bm_real, reshaped_transposed_imag) + tf.matmul(bm_imag, reshaped_transposed_real)

                        # term_dd_real = tf.reshape(tf.transpose(tf.reshape(term_dd_real, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                        # term_dd_imag = tf.reshape(tf.transpose(tf.reshape(term_dd_imag, [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])

                        term_dd_real = tf.reshape(tf.transpose(tf.reshape(tf.matmul(tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)  ], [-1, l2+1, l1+1]), reshaped_transposed_real) - tf.matmul(tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)+1], [-1, l2+1, l1+1]), reshaped_transposed_imag), [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])
                        term_dd_imag = tf.reshape(tf.transpose(tf.reshape(tf.matmul(tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)  ], [-1, l2+1, l1+1]), reshaped_transposed_imag) + tf.matmul(tf.reshape(bessel_matrices[2*(l1*(self.L+1)+l2)+1], [-1, l2+1, l1+1]), reshaped_transposed_real), [-1, self.P, l2+1, input_channels]), [0,2,1,3]),[-1, l2+1, self.P*input_channels])

                        
                        
                        if len(first_rotation_and_translation) < 2*(l2+1):
                            first_rotation_and_translation.append(term_dd_real)
                            first_rotation_and_translation.append(term_dd_imag)
                        else:
                            first_rotation_and_translation[2*l2] += term_dd_real
                            first_rotation_and_translation[2*l2+1] += term_dd_imag
                
                shifted = []
                for l in range(self.L+1):
                    # second_rotation_real  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l]) + tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1])
                    # second_rotation_imag  = tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l])
                    # second_rotation_real += tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1])
                    # second_rotation_imag += 0-tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l])
                    
                        
                    # term_real = input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_real,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    # term_imag = input[2*l+1] +tf.reshape(tf.matmul(E_to_N, tf.reshape(second_rotation_imag,  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])

                    term_real = input[2*l] + tf.reshape(tf.matmul(E_to_N, tf.reshape(tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l]) + tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l+1]) + tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l+1]),  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                    term_imag = input[2*l+1] +tf.reshape(tf.matmul(E_to_N, tf.reshape(tf.matmul(second_slater_matrices[4*l], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+1], first_rotation_and_translation[2*l]) - tf.matmul(second_slater_matrices[4*l+2], first_rotation_and_translation[2*l+1]) - tf.matmul(second_slater_matrices[4*l+3], first_rotation_and_translation[2*l]),  [-1, (l+1)*self.P*input_channels])), [-1, l+1, self.P, input_channels])
                        
                    shifted.append( term_real)
                    shifted.append( term_imag)
            else:
                shifted = input 
                    
            convoluted, biases = self.full_convolution_in_6D(shifted, output_channels = output_channel, name = name + '_' + str(g) + '_convolution', with_bias = True)
            gamma = tf.Variable(1.0, trainable=True, name=name+"_normalization_gamma")
            beta = tf.Variable(0.01, trainable=True, name=name+"_normalization_beta")
            with tf.name_scope(name):
                tf.summary.scalar(name+"_normalization_gamma", gamma)
                tf.summary.scalar(name+"_normalization_beta", beta)
            normalized_tensor = self.normalization2(convoluted, gamma, beta, name = name + '_' + str(g) + '_normalization')
            normalized_biases = self.normalization2(biases, gamma, beta, name = name + '_' + str(g) + '_normalization_bias')
            output_group = self.function_activation3( normalized_tensor, normalized_biases,  name = name + '_' + str(g) + '_function_activation3')
            for l in range(self.L+1):
                output_group_degree_real = output_group[2*l]/self.G
                output_group_degree_imag = output_group[2*l+1]/self.G
                if len(output) < 2*(self.L+1):
                    output.append(output_group_degree_real)
                    output.append(output_group_degree_imag)
                    
                else:
                    output[2*l] += output_group_degree_real
                    output[2*l+1] += output_group_degree_imag


                
            
        return output

    
    
    
    def retyper_block(self, input, output_channel = 10, internal_layer = True, name = 'retyper_block'):    
        retyped_input = self.retyper(input, output_channels=output_channel, with_bias=internal_layer, name = name + 'retyper')
        if internal_layer:
            norm_input = self.normalization(retyped_input, name = name + 'normalization')
            output = self.function_activation(norm_input, name = name + 'function_activation')
        else:
            output = retyped_input
        return output

    def full_kernel_convolution_block(self, input, output_channel = 10, name = 'full_kernel_convolution_block'):
        convoluted = self.full_kernel_convolution(input, output_channels=output_channel, name = name + 'full_kernel_convolution')
        normalized = self.normalization(convoluted,  name = name + 'normalization')
        output = self.function_activation(normalized, name = name + 'function_activation')
        return output
    
    def function_to_value(self, input, name = 'embedding'):
        output = 0.0
        input_channels = input[0].get_shape().as_list()[-1]
        for l in range(self.L+1):
            weights_degree_real = _weight_variable("weights_" + name + "_" + str(l) + "_real", [1, l+1, self.P, input_channels])
            weights_degree_imag = _weight_variable("weights_" + name + "_" + str(l) + "_imag", [1, l+1, self.P, input_channels])
            temp_term = np.ones((1, l+1, self.P, 1))
            temp_term[:,0,:,:] = 0
            temp_term = tf.convert_to_tensor(temp_term, dtype = tf.float32)
            weights_degree_imag = weights_degree_imag*temp_term
            with tf.name_scope(name + "_" + str(l)):
                tf.summary.histogram("weights_" + name + "_" + str(l) + "_real", weights_degree_real )
                tf.summary.histogram("weights_" + name + "_" + str(l) + "_imag", weights_degree_imag )
            output += tf.reduce_mean(tf.reduce_mean(weights_degree_real*input[2*l], axis = 2), axis = 1)
            output += -tf.reduce_mean(tf.reduce_mean(weights_degree_imag*input[2*l+1], axis = 2), axis = 1)
            if l > 0:
                output += tf.reduce_mean(tf.reduce_mean(weights_degree_real*input[2*l], axis = 2), axis = 1)
                output += -tf.reduce_mean(tf.reduce_mean(weights_degree_imag*input[2*l+1], axis = 2), axis = 1)

        
        
        return output


    
    def classical_graph_convolution(self, input, adjacency_matrices, output_channel = 10, activation = 'lrelu', name = 'graph_conv' ):
        output = 0.0
        input_channels = input.get_shape().as_list()[-1]
        for g, adjacency_matrix in enumerate(adjacency_matrices):
            weights_group = _weight_variable("weights_" + name + "_" + str(g), [input_channels, output_channel])
            with tf.name_scope(name):
                tf.summary.histogram("weights_" + name + "_" + str(g), weights_group )
            if activation == 'lrelu':
                output += lrelu(tf.matmul(tf.matmul(adjacency_matrix, input), weights_group), leak = LEAK)/self.G
            elif activation == 'tanh':
                output += tf.tanh(tf.matmul(tf.matmul(adjacency_matrix, input), weights_group))/self.G
        return output


    def functional_block(self, input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = 'fun_part'):
        
        print(self.params_conv.layers)
        for l_i, layer in enumerate(self.params_conv.layers):
            if layer.split('_')[0] == 'agg':
                input = self.aggregation_block(input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = name + '_agg_block_'+str(l_i))
            elif layer.split('_')[0] == 'a6D':
                input = self.aggregation_nc_block(input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, name = name + '_agg_nc_block_'+str(l_i))
            elif layer.split('_')[0] == 'ret':
                output_channel = int(layer.split('_')[1])
                input = self.retyper_block(input, output_channel = output_channel, name = name + '_retyper_'+str(l_i))
            elif layer.split('_')[0] == 'full':
                output_channel = int(layer.split('_')[1])
                input = self.full_kernel_convolution_block(input, output_channel = output_channel, name = name + '_full_kernel_convolution_'+str(l_i))
            elif layer.split('_')[0] == 'afk':
                output_channel = int(layer.split('_')[1])
                input = self.full_kernel_aggregation_block(input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, output_channel = output_channel, name = name + '_full_kernel_aggregation_'+str(l_i))
            elif layer.split('_')[0] == 'a6f':
                output_channel = int(layer.split('_')[1])
                input = self.full_kernel_aggregation_nc_block(input, groups_of_edges, first_slater_matrices, bessel_matrices, second_slater_matrices, output_channel = output_channel, do_aggregation = (l_i != 0), name = name + '_full_kernel_aggregation_nc_'+str(l_i))
            for l in range(self.L+1):
                with tf.name_scope(name + "_" + str(l_i)+  "_" + str(l)):
                    tf.summary.histogram("outputs_" + name + "_" + str(l) + "_real", input[2*l] )
                    tf.summary.histogram("outputs_" + name + "_" + str(l) + "_imag", input[2*l+1] )
        return input
    
    def vector_block(self, input, adjacency_matrices, name = 'vector_part'):
    
        for l_i, channel in enumerate(self.params_conv.output_channels):
            input = self.classical_graph_convolution(input, adjacency_matrices, output_channel = channel, name = 'graph_conv_' + str(l_i))
        return input


    




    


    def compute_loss(self, predicted_directions, directions):
        """
        Compute loss, l2 distance
        :param scores: List of predicted scores for batch
        :param cad_score: List of ground truth scores
        :return: list of l2 distance between predicted scores and gt scores
        """
        print(predicted_directions.get_shape())
        print(directions.get_shape())
        return tf.reduce_sum(tf.square(predicted_directions - directions), axis = 1, name='loss')


    def train(self, loss, learning_rate):
        """
        Train operations when building graph. With adam optimizer
        :param loss: output of loss function
        :param learning_rate: Learning rate
        :return: training operation output of Optimizer.minimize()
        """
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(tf.reduce_mean(loss), global_step=global_step, name='train_op')

        return train_op


        