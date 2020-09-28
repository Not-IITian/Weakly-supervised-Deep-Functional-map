import numpy as np
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def penalty_bijectivity(C_est_AB, C_est_BA):
    
    return tf.nn.l2_loss(tf.subtract(tf.matmul(C_est_AB, C_est_BA),tf.eye(tf.shape(C_est_AB)[1])
                                    ))


def penalty_ortho(C_est):
   
    return tf.nn.l2_loss(tf.subtract(tf.matmul(tf.transpose(C_est, perm=[0, 2, 1]),
                                       C_est),tf.eye(tf.shape(C_est)[1])))


def penalty_laplacian_commutativity(C_est, source_evals, target_evals):
    
    # Quicker and less memory than taking diagonal matrix
    eig1 = tf.einsum('abc,ac->abc', C_est, source_evals)
    eig2 = tf.einsum('ab,abc->abc', target_evals, C_est)

    return tf.nn.l2_loss(tf.subtract(eig2, eig1))

    
def pointwise_corr_layer(C_est, source_evecs, target_evecs_trans, source_dist_map, target_dist_map):
    
    P = tf.matmul(tf.matmul(source_evecs, C_est), target_evecs_trans)
    P = tf.abs(P)

    P_norm = tf.nn.l2_normalize(P, dim=1, name='soft_correspondences')  
    #unsupervised loss calculation
    avg_distance_on_model_after_map = tf.einsum('kmn,kmi,knj->kij', source_dist_map, tf.pow(P_norm,2), tf.pow(P_norm,2))  vertices on the model
    avg_distortion_after_map = avg_distance_on_model_after_map - target_dist_map
    unsupervised_loss = tf.nn.l2_loss(avg_distortion_after_map)
    unsupervised_loss /= tf.to_float(tf.shape(P)[0] * tf.shape(P)[2] * tf.shape(P)[2]) 

    return P_norm, unsupervised_loss


    
def sup_penalty_surreal(C_est, source_evecs, target_evecs):
    """
    Args: full_source_evecs and full_target_evecs are over batch..
    """    
    fmap = tf.matrix_solve_ls(tf.transpose(target_evecs,[0,2,1]),tf.transpose(source_evecs,[0,2,1]))            
    return tf.nn.l2_loss(C_est - fmap)

def func_map_layer(C_est_AB, C_est_BA,source_evecs, source_evecs_trans, source_evals,
                target_evecs, target_evecs_trans, target_evals, F, G, source_dist,target_dist):
    
    alpha = 1	#10**3  # Bijectivity
    beta = 1#10**3   # Orthogonality
    gamma = .001#1      # Laplacian commutativity
    delta = 0 # Descriptor preservation via commutativity

    E1 = penalty_bijectivity(C_est_AB, C_est_BA) +penalty_bijectivity(C_est_BA, C_est_AB))/2

    E2 = penalty_ortho(C_est_AB)  + penalty_ortho(C_est_BA))/2
    #E5 = 0
    E4=0
    E3 = (penalty_laplacian_commutativity(C_est_AB,source_evals,target_evals) 
          +penalty_laplacian_commutativity(C_est_BA, target_evals, source_evals))/2

    E5=0
    #E5 =  (sup_penalty_surreal(C_est_AB, F, G) + sup_penalty_surreal(C_est_BA, G,F))/2
    #_,E5 =pointwise_corr_layer(C_est_AB, source_evecs, target_evecs_trans, source_dist, target_dist)
    loss = tf.reduce_mean(alpha * E1 + beta * E2 + gamma * E3 + delta * E4 + E5)    
     #check this..
    loss /= tf.to_float(tf.shape(C_est_AB)[1] * tf.shape(C_est_AB)[0])

    C_est_AB = tf.reshape(C_est_AB,
        [FLAGS.batch_size, tf.shape(C_est_AB)[1], tf.shape(C_est_AB)[2], 1])
    tf.summary.image("Estimated_FuncMap_AB", C_est_AB, max_outputs=1)

    C_est_BA = tf.reshape(C_est_BA, [FLAGS.batch_size, tf.shape(C_est_BA)[1], tf.shape(C_est_BA)[2], 1])
    tf.summary.image("Estimated_FuncMap_BA", C_est_BA, max_outputs=1)

    return loss, E1, E2, E3, E4

