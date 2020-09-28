#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tensorflow as tf
import numpy as np
import tf_util

from loss import *
from pointnet_util import pointnet_sa_module, pointnet_fp_module
flags= tf.app.flags
FLAGS=flags.FLAGS


def res_layer(x_in, dims_out, scope, phase):    
    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(x_in,dims_out, activation_fn=None,scope='dense_1')
        x = tf.contrib.layers.batch_norm( x, center=True, scale=True, is_training=phase, variables_collections=["batch_norm_non_trainable_variables_collection"],
                                            scope='bn_1')
        x = tf.nn.relu(x, 'relu')
        x = tf.contrib.layers.fully_connected( x,dims_out,activation_fn=None,scope='dense_2')
        x = tf.contrib.layers.batch_norm( x, center=True, scale=True, is_training=phase,variables_collections=["batch_norm_non_trainable_variables_collection"],
                                            scope='bn_2')
        # If dims_out change, modify input via linear projection
        # (as suggested in resNet)
        if not x_in.get_shape().as_list()[-1] == dims_out:
            x_in = tf.contrib.layers.fully_connected( x_in, dims_out, activation_fn=None, scope='projection')
        x += x_in
        return tf.nn.relu(x)

def solve_ls(A, B):
    # Transpose input matrices
    At = tf.transpose(A, [0, 2, 1])
    Bt = tf.transpose(B, [0, 2, 1])

    # Solve C via least-squares
    Ct_est = tf.matrix_solve_ls(At, Bt)
    #Ct_est = tf.matrix_solve_ls(At, Bt, l2_regularizer = 0.000001)
    C_est = tf.transpose(Ct_est, [0, 2, 1], name='C_est')

    # Calculate error for safeguarding
    safeguard_inverse = tf.nn.l2_loss(tf.matmul(At, Ct_est) - Bt)
    safeguard_inverse /= tf.to_float(tf.reduce_prod(tf.shape(A)))

    return C_est, safeguard_inverse

def get_model(phase, pc1, pc2, source_evecs, source_evecs_trans, source_evals,
                target_evecs, target_evecs_trans, target_evals, src_dist=None, tar_dist=None,bn_decay=None):
    dims=FLAGS.dims
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """    
    batch_size = pc1.get_shape()[0].value
    print(batch_size)
    num_point = pc1.get_shape()[1].value
    print(num_point)
    end_points = {}
    l0_xyz_s = pc1
    l0_xyz_t = pc2
    l0_points_s,l0_points_t = (None,None)
    
    end_points['l0_xyz_s'] = l0_xyz_s
    end_points['l0_xyz_t'] = l0_xyz_t
    # Layer 1
    with tf.variable_scope('layer_1') as scope:
        l1_xyz_s, l1_points_s, l1_indices_s = pointnet_sa_module(l0_xyz_s, l0_points_s, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
        scope.reuse_variables()
        l1_xyz_t, l1_points_t, l1_indices_t = pointnet_sa_module(l0_xyz_t, l0_points_t, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
    
    with tf.variable_scope('layer_2') as scope:           
        l2_xyz_s, l2_points_s, l2_indices_s = pointnet_sa_module(l1_xyz_s, l1_points_s, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
        scope.reuse_variables()
        l2_xyz_t, l2_points_t, l2_indices_t = pointnet_sa_module(l1_xyz_t, l1_points_t, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
        
    with tf.variable_scope('layer_3') as scope:                   
        l3_xyz_s, l3_points_s, l3_indices_s = pointnet_sa_module(l2_xyz_s, l2_points_s, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
        scope.reuse_variables()
        l3_xyz_t, l3_points_t, l3_indices_t = pointnet_sa_module(l2_xyz_t, l2_points_t, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
    
    with tf.variable_scope('layer_4') as scope:     
        l4_xyz_s, l4_points_s, l4_indices_s = pointnet_sa_module(l3_xyz_s, l3_points_s, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)
        scope.reuse_variables()
        l4_xyz_t, l4_points_t, l4_indices_t = pointnet_sa_module(l3_xyz_t, l3_points_t, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=phase, bn_decay=bn_decay, scope=scope)

    # Feature Propagation layers
    with tf.variable_scope('fa_layer1') as scope:
        l3_points_s = pointnet_fp_module(l3_xyz_s, l4_xyz_s, l3_points_s, l4_points_s, [256,256], phase, bn_decay, scope=scope)
        scope.reuse_variables()
        l3_points_t = pointnet_fp_module(l3_xyz_t, l4_xyz_t, l3_points_t, l4_points_t, [256,256], phase, bn_decay, scope=scope)
    
    with tf.variable_scope('fa_layer2') as scope:        
        l2_points_s = pointnet_fp_module(l2_xyz_s, l3_xyz_s, l2_points_s, l3_points_s, [256,256], phase, bn_decay, scope=scope)
        scope.reuse_variables()
        l2_points_t = pointnet_fp_module(l2_xyz_t, l3_xyz_t, l2_points_t, l3_points_t, [256,256], phase, bn_decay, scope=scope)
    
    with tf.variable_scope('fa_layer3') as scope:        
        l1_points_s = pointnet_fp_module(l1_xyz_s, l2_xyz_s, l1_points_s, l2_points_s, [256,128], phase, bn_decay, scope=scope)
        scope.reuse_variables()
        l1_points_t = pointnet_fp_module(l1_xyz_t, l2_xyz_t, l1_points_t, l2_points_t, [256,128], phase, bn_decay, scope=scope)
    
    with tf.variable_scope('fa_layer4') as scope:        
        l0_points_s = pointnet_fp_module(l0_xyz_s, l1_xyz_s, l0_points_s, l1_points_s, [128,128,128], phase, bn_decay, scope=scope)
        scope.reuse_variables()
        l0_points_t = pointnet_fp_module(l0_xyz_t, l1_xyz_t, l0_points_t, l1_points_t, [128,128,128], phase, bn_decay, scope=scope)
    net = {}    
    for i_layer in range(FLAGS.num_fclayers):
        with tf.variable_scope("layer_%d" % i_layer) as scope:
            if i_layer == 0:
                net['fclayer_%d_s' % i_layer] = res_layer(l0_points_s, dims_out=128, scope=scope,phase=phase)
                scope.reuse_variables()
                net['fclayer_%d_t' % i_layer] = res_layer(l0_points_t, dims_out=128, scope=scope,phase=phase)
            else:
                net['fclayer_%d_s' % i_layer] = res_layer(net['fclayer_%d_s' % (i_layer-1)], dims_out=int(dims[i_layer]),scope=scope, phase=phase)
                scope.reuse_variables()
                net['fclayer_%d_t' % i_layer] = res_layer(net['fclayer_%d_t' % (i_layer-1)],dims_out=int(dims[i_layer]),scope=scope,phase=phase)
                                        
    #  Project output features on the shape Laplacian eigen functions    
    layer_C_est = i_layer + 1   # Grab current layer index
    F = net['fclayer_%d_s' % (layer_C_est-1)]
    A = tf.matmul(source_evecs_trans, F)
    net['A'] = A
    
    G = net['fclayer_%d_t' % (layer_C_est-1)]
    B = tf.matmul(target_evecs_trans, G)
    net['B'] = B
    #  FM-layer: evaluate C_est
    net['C_est_AB'], safeguard_inverse = solve_ls(A, B)
    net['C_est_BA'], safeguard_inverse = solve_ls(B, A)

    #  Evaluate loss without any ground-truth or geodesic distance matrix
    with tf.variable_scope("func_map_loss"):
        net_loss, E1, E2, E3, E4 = func_map_layer(net['C_est_AB'], net['C_est_BA'], source_evecs, source_evecs_trans, source_evals,target_evecs, target_evecs_trans, target_evals,A, B, src_dist,tar_dist)

    tf.summary.scalar('net_loss_Bijectivity', E1)
    tf.summary.scalar('net_loss_Orthogonality', E2)
    tf.summary.scalar('net_loss_LaplacianCommutativity', E3)
   
    tf.summary.scalar('net_loss', net_loss)
    merged = tf.summary.merge_all()
    return net_loss, safeguard_inverse, net, end_points, net['C_est_AB'], merged

def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """    
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
