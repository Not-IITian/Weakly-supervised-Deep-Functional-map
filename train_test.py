

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import scipy.io as sio
from model import *
#!/usr/bin/env python3
# -*- coding: utf-8 -*

flags = tf.app.flags
FLAGS = flags.FLAGS
# Training parameterss
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 8, 'batch size.')
# Architecture parameters
flags.DEFINE_integer('num_fclayers', 2, 'network depth') 
flags.DEFINE_integer('num_evecs', 30, "number of eigenvectors used for representation")

flags.DEFINE_string('targets_dir', '../../Unsupervised_FMnet/Shapes/Surreal/MAT/','directory with shapes')  
flags.DEFINE_string('feat_dir', '../../Unsupervised_FMnet/Shapes/Surreal/MAT/','directory with shapes')  
flags.DEFINE_string('files_name', 'surreal_', 'name common to all the shapes')

flags.DEFINE_string('te_dir_F', '../../Unsupervised_FMnet/Shapes/Faust_r_aligned/MAT/','directory with shapes')  
flags.DEFINE_string('feat_dir_te_F', '../../Unsupervised_FMnet/Shapes/Faust_r_aligned/faust_mat/','directory with shapes')  
flags.DEFINE_string('files_name_F', '', 'name common to all the shapes')

flags.DEFINE_string('te_dir_S', '../../Unsupervised_FMnet/Shapes/Scape_r_aligned/MAT/','directory with shapes')  
flags.DEFINE_string('feat_dir_te_S', '../../Unsupervised_FMnet/Shapes/Scape_r_aligned/scape_mat/','directory with shapes')  
flags.DEFINE_string('files_name_S', '', 'name common to all the shapes')

flags.DEFINE_integer('max_train_iter', 10000, '')
flags.DEFINE_integer('num_vertices', 3000, '') 
flags.DEFINE_integer('save_summaries_secs', 500, '') 
flags.DEFINE_integer('save_model_secs', 500, '')
flags.DEFINE_string('log_dir_', 'Training/SCAPE_r_aligned/pointnet_fmnet_b8_lr_-4_30evec_.001_6k_2fc_aligned_tr-Sur_100_rand_sup_both_sides_1_1.001_sbb-/',
                    'directory to save models and results')
flags.DEFINE_string('matches_dir_F', './Matches/FAUST_r_aligned/ptnet_surreal_80rand_3k_F_30_1_1_.001_b8_both_te_20/','directory to save models and results')
flags.DEFINE_string('matches_dir_S', './Matches/SCAPE_r_aligned/ptnet_surreal_80_rand_3k_S_30_1_1_.001_b8_both_te_20/','directory to save models and results')
flags.DEFINE_integer('dim_', 128,'')
flags.DEFINE_integer('decay_step', 200000, help='Decay step for lr decay [default: 200000]')

flags.DEFINE_float('decay_rate', 0.7, help='Decay rate for lr decay [default: 0.7]')

# Globals
dim_=FLAGS.dim_   
flags.DEFINE_list('dims', [dim_,dim_,dim_,dim_, dim_, dim_, dim_], '')      
       
dim_1_layer = int(FLAGS.dims[0])
flags.DEFINE_integer('dim_shot', dim_1_layer, '')  
no_layers = FLAGS.num_fclayers

last_layer = int(FLAGS.dims[no_layers-1])
flags.DEFINE_integer('dim_out', last_layer, '') 

vert_dir = FLAGS.feat_dir
vert_dir_te_S = FLAGS.feat_dir_te_S
vert_dir_te_F = FLAGS.feat_dir_te_F
n_tr = 100

#train_subjects = list(range(n_tr))
train_subjects = np.random.choice(1000,n_tr)
test_subjects_F,test_subjects_S = (range(80,100),range(52,70))
main_dir = FLAGS.targets_dir
files_name =FLAGS.files_name
te_files_name_F = FLAGS.files_name_F
te_files_name_S = FLAGS.files_name_S
#te_files_name = ''
test_dir_F=FLAGS.te_dir_F
test_dir_S=FLAGS.te_dir_S

 
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate 
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99 
BATCH_SIZE = FLAGS.batch_size

def get_input_pair(batch_size, num_vertices, dataset):
    
    batch_input = {
        'source_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
        'target_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
        'source_evecs_trans': np.zeros((batch_size,FLAGS.num_evecs,num_vertices)),
        'target_evecs_trans': np.zeros((batch_size,FLAGS.num_evecs,num_vertices)),
        'source_shot': np.zeros((batch_size, num_vertices, 3)),
        'target_shot': np.zeros((batch_size, num_vertices, 3)),
        'source_evals': np.zeros((batch_size, FLAGS.num_evecs)),
        'target_evals': np.zeros((batch_size, FLAGS.num_evecs))
                   }       
    for i_batch in range(batch_size):                      
        i_source = train_subjects[np.random.choice(range(n_tr))]
        i_target = train_subjects[np.random.choice(range(n_tr))]          
        
        batch_input_ = get_pair_from_ram(i_target, i_source, dataset)
        
        batch_input_['source_labels'] = range(np.shape(batch_input_['source_evecs'])[0])
        batch_input_['target_labels'] = range(np.shape(batch_input_['target_evecs'])[0])
        
        joint_lbls = np.intersect1d(batch_input_['source_labels'],batch_input_['target_labels'])
        #print(joint_lbls)
        joint_labels_source = np.random.permutation(joint_lbls)[:num_vertices]
        joint_labels_target = np.random.permutation(joint_lbls)[:num_vertices]

        ind_dict_source = {value: ind for ind, value in enumerate(batch_input_['source_labels'])}
        ind_source = [ind_dict_source[x] for x in joint_labels_source]

        ind_dict_target = {value: ind for ind, value in enumerate(batch_input_['target_labels'])}
        ind_target = [ind_dict_target[x] for x in joint_labels_target]
        
        message = "number of indices must be equal"
        assert len(ind_source) == len(ind_target), message
        
        evecs_full = batch_input_['source_evecs']
        #print(evecs_full.shape)
        evecs= evecs_full[ind_source, :]
        evecs_trans = batch_input_['source_evecs_trans'][:, ind_source]
        shot = batch_input_['source_shot'][ind_source, :]
        #print(batch_input_['target_shot'].shape)                
        evals = [item for sublist in batch_input_['source_evals'] for item in sublist] # what?
        batch_input['source_evecs'][i_batch] = evecs
        batch_input['source_evecs_trans'][i_batch] = evecs_trans
        batch_input['source_shot'][i_batch] = shot
        batch_input['source_evals'][i_batch] = evals

        evecs = batch_input_['target_evecs'][ind_target, :]
        evecs_trans = batch_input_['target_evecs_trans'][:, ind_target]
        shot = batch_input_['target_shot'][ind_target, :]
        evals = [item for sublist in batch_input_['target_evals'] for item in sublist]
        batch_input['target_evecs'][i_batch] = evecs
        batch_input['target_evecs_trans'][i_batch] = evecs_trans
        batch_input['target_shot'][i_batch] = shot
        batch_input['target_evals'][i_batch] = evals
    return batch_input




def get_input_pair_test(i_target, i_source, sub_):
    batch_input = {}
    batch_input_ = get_pair_from_ram(i_target, i_source, sub_)
    
    evecs = batch_input_['source_evecs']
    evecs_trans = batch_input_['source_evecs_trans']
    shot = batch_input_['source_shot']
    evals = [item for sublist in batch_input_['source_evals'] for item in sublist]
    batch_input['source_evecs'] = evecs
    batch_input['source_evecs_trans'] = evecs_trans
    batch_input['source_shot'] = shot
    batch_input['source_evals'] = evals

    evecs = batch_input_['target_evecs']
    evecs_trans = batch_input_['target_evecs_trans']
    shot = batch_input_['target_shot']
    evals = [item for sublist in batch_input_['target_evals'] for item in sublist]
    batch_input['target_evecs'] = evecs
    batch_input['target_evecs_trans'] = evecs_trans
    batch_input['target_shot'] = shot
    batch_input['target_evals'] = evals
    return batch_input

def get_pair_from_ram(i_target, i_source, sub_):
    
    input_data = {}
    if sub_ == 'train':
        targets_= targets_train
    elif sub_== 'te_F':
        targets_=targets_test_F
    else: 
        targets_=targets_test_S
            
    evecs = targets_[i_source]['target_evecs']
    evecs_trans = targets_[i_source]['target_evecs_trans']
    shot = targets_[i_source]['target_shot']
    evals = targets_[i_source]['target_evals']
    input_data['source_evecs'] = evecs
    input_data['source_evecs_trans'] = evecs_trans
    input_data['source_shot'] = shot
    input_data['source_evals'] = evals
    input_data.update(targets_[i_target])
       
    return input_data

def load_targets_to_ram():
    global targets_train,targets_test_F,targets_test_S
    targets_train,targets_test_F,targets_test_S = ({},{},{}) 
    
    targets_train = load_subs(train_subjects, main_dir, vert_dir,files_name)
    targets_test_F= load_subs(test_subjects_F,test_dir_F, vert_dir_te_F, te_files_name_F)
    targets_test_S =load_subs(test_subjects_S,test_dir_S, vert_dir_te_S,te_files_name_S)
    
def load_subs(subjects_list, dir_name,v_dir,f_name): 
    targets = {}    
    
    for i_target in subjects_list:             
        target_file = dir_name +f_name +'%.4d.mat' % (i_target)
        vert_file = v_dir +f_name +'%.4d.mat' % (i_target)
        #print(vert_file)
        input_data = sio.loadmat(target_file)        
        evecs = input_data['target_evecs'][:, 0:FLAGS.num_evecs]
        evecs_trans = input_data['target_evecs_trans'][0:FLAGS.num_evecs,:]
        evals = input_data['target_evals'][0:FLAGS.num_evecs]        
        input_data['target_evecs'] = evecs
        input_data['target_evecs_trans'] = evecs_trans
        input_data['target_evals'] = evals 
        p_feat = sio.loadmat(vert_file)
        input_data['target_shot'] =[]        
        input_data['target_shot'] = p_feat['VERT']             
        targets[i_target] = input_data
        
    return targets


def get_bn_decay(batch):    
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay 
   
def run_training():
    
    print('log_dir=%s' % FLAGS.log_dir_)
    if not os.path.isdir(FLAGS.log_dir_):
        os.makedirs(FLAGS.log_dir_)
    
    print('building graph...')
    
    with tf.Graph().as_default():
        # Set placeholders for inputs        
        source_shot = tf.placeholder(tf.float32,shape=(None, None, 3),name='source_shot')
        target_shot = tf.placeholder(tf.float32, shape=(None, None, 3),name='target_shot')
        
        source_evecs = tf.placeholder(tf.float32, shape=(None, None, FLAGS.num_evecs), name='source_evecs')
        source_evecs_trans = tf.placeholder(tf.float32,shape=(None, FLAGS.num_evecs, None),name='source_evecs_trans')
        source_evals = tf.placeholder(tf.float32,shape=(None, FLAGS.num_evecs),name='source_evals')
        target_evecs = tf.placeholder(tf.float32,shape=(None, None, FLAGS.num_evecs),name='target_evecs')
        target_evecs_trans = tf.placeholder(tf.float32,shape=(None, FLAGS.num_evecs, None),name='target_evecs_trans')
        target_evals = tf.placeholder(tf.float32,shape=(None, FLAGS.num_evecs),name='target_evals')
        # train\test switch flag
        phase = tf.placeholder(dtype=tf.bool, name='phase') 
        
        #is_training_pl = tf.placeholder(tf.bool, shape=())
        #print (is_training_pl)
        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)
              
        net_loss, safeguard_inverse, net,end_points, C, merged = get_model(phase, source_shot, target_shot, source_evecs, source_evecs_trans,
                                            source_evals, target_evecs, target_evecs_trans, target_evals, bn_decay)
        
        summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(net_loss,global_step=global_step, aggregation_method=2)
            
            #saver = tf.train.Saver(max_to_keep=40)
        saver = tf.train.Saver(tf.global_variables())   
        
        sv = tf.train.Supervisor(logdir=FLAGS.log_dir_,init_op=tf.global_variables_initializer(),local_init_op=tf.local_variables_initializer(),
                                global_step=global_step,save_summaries_secs=FLAGS.save_summaries_secs,
                                save_model_secs=FLAGS.save_model_secs,summary_op=None,saver=saver)  
        
        writer = sv.summary_writer
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print('starting session...')
        iteration = 0
            
        with sv.managed_session(config=config) as sess:
            print('loading data to ram...')
            load_targets_to_ram()
            
            print('starting training loop...')
            while not sv.should_stop() and iteration < FLAGS.max_train_iter:
                
                iteration += 1
                start_time = time.time()
                input_data = get_input_pair(FLAGS.batch_size, FLAGS.num_vertices, 'train')
                
                feed_dict = {phase: False, source_shot: input_data['source_shot'], target_shot: input_data['target_shot'],
                    source_evecs: input_data['source_evecs'], source_evecs_trans: input_data['source_evecs_trans'],
                    source_evals: input_data['source_evals'], target_evecs: input_data['target_evecs'],
                    target_evecs_trans: input_data['target_evecs_trans'], target_evals: input_data['target_evals']}

                summaries, step, my_loss, safeguard, _ = sess.run([merged, global_step, net_loss, safeguard_inverse, train_op],
                  feed_dict=feed_dict)
                
                writer.add_summary(summaries, step)
                summary_ = sess.run(summary)
                writer.add_summary(summary_, step)
                duration = time.time() - start_time
                print('train - step %d: loss = %.2f (%.3f sec)'% (step, my_loss, duration))
                
                if iteration%1000==0:                    
                    for i_source in range(52,69):     
                        for i_target in range(i_source+1,70):
                            
                            t = time.time()
                            
                            input_data = get_input_pair_test(i_target, i_source, 'te_S')
                            source_evecs_ = input_data['source_evecs'][:, 0:FLAGS.num_evecs]
                            target_evecs_ = input_data['target_evecs'][:, 0:FLAGS.num_evecs]
        
                            feed_dict = {
                                phase: False,
                                source_shot: [input_data['source_shot']],
                                target_shot: [input_data['target_shot']],
                                source_evecs: [input_data['source_evecs']],
                                source_evecs_trans: [input_data['source_evecs_trans']],
                                source_evals: [input_data['source_evals']],
                                target_evecs: [input_data['target_evecs']],
                                target_evecs_trans: [input_data['target_evecs_trans']],
                                target_evals: [input_data['target_evals']]
                                }
                
                            C_est_ = sess.run([C], feed_dict=feed_dict)                            
                            Ct = np.squeeze(C_est_).T #Keep transposed
                
                            kdt = cKDTree(np.matmul(source_evecs_, Ct))
                            
                            dist, indices = kdt.query(target_evecs_, n_jobs=-1)
                            indices = indices + 1
                
                            print("Computed correspondences for pair: %s, %s." % (i_source, i_target) +
                                  " Took %f seconds" % (time.time() - t))                
                            params_to_save = {}
                            params_to_save['matches'] = indices
                            #params_to_save['C'] = Ct.T
                            # For Matlab where index start at 1  
                            new_dir = FLAGS.matches_dir_S + '%.3d-' % iteration + '/'
                            
                            if not os.path.isdir(new_dir):
                                print('matches_dir=%s' % new_dir)        
                                os.makedirs(new_dir)
                            
                            sio.savemat(new_dir  + '%.4d-' % i_source  + '%.4d.mat' % i_target, params_to_save)
                    
                    for i_source in range(80,99):     
                        for i_target in range(i_source+1,100):
                            
                            t = time.time()
                            
                            input_data = get_input_pair_test(i_target, i_source, 'te_F')
                            source_evecs_ = input_data['source_evecs'][:, 0:FLAGS.num_evecs]
                            target_evecs_ = input_data['target_evecs'][:, 0:FLAGS.num_evecs]
        
                            feed_dict = {
                                phase: False,
                                source_shot: [input_data['source_shot']],
                                target_shot: [input_data['target_shot']],
                                source_evecs: [input_data['source_evecs']],
                                source_evecs_trans: [input_data['source_evecs_trans']],
                                source_evals: [input_data['source_evals']],
                                target_evecs: [input_data['target_evecs']],
                                target_evecs_trans: [input_data['target_evecs_trans']],
                                target_evals: [input_data['target_evals']]
                                }
                
                            C_est_ = sess.run([C], feed_dict=feed_dict)                            
                            Ct = np.squeeze(C_est_).T #Keep transposed
                
                            kdt = cKDTree(np.matmul(source_evecs_, Ct))
                            
                            dist, indices = kdt.query(target_evecs_, n_jobs=-1)
                            indices = indices + 1
                
                            print("Computed correspondences for pair: %s, %s." % (i_source, i_target) +
                                  " Took %f seconds" % (time.time() - t))                
                            params_to_save = {}
                            params_to_save['matches'] = indices
                            #params_to_save['C'] = Ct.T
                            # For Matlab where index start at 1  
                            new_dir = FLAGS.matches_dir_F + '%.3d-' % iteration + '/'
                            
                            if not os.path.isdir(new_dir):
                                print('matches_dir=%s' % new_dir)        
                                os.makedirs(new_dir)
                            
                            sio.savemat(new_dir  + '%.4d-' % i_source  + '%.4d.mat' % i_target, params_to_save)
                    

def main(_):
    import time
    start_time = time.time()
    run_training()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()

                
              
