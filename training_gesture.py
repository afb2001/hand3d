from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys

from nets.PosePriorNetwork import PosePriorNetwork
from nets.GestureCommandNetwork import GestureCommandNetwork
from data.BinaryDbReader import BinaryDbReader
from utils.general import LearningRateScheduler

# Chose which variant to evaluate
# VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
# VARIANT = 'proposed'
VARIANT = 'gesture'

# training parameters
train_para = {'lr': [1e-5, 1e-6],
              'lr_iter': [60000],
              'max_iter': 80000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_gesture'}

# get dataset # AFB - assume this is fine for now
dataset = BinaryDbReader(mode='training',
                         batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
                         coord_uv_noise=True, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

# build network graph
data = dataset.get()

# build network
# net = PosePriorNetwork(VARIANT)
net = GestureCommandNetwork()

# feed trough network
evaluation = tf.placeholder_with_default(True, shape=())
# _, coord3d_pred, R = net.inference(data['scoremap'], data['hand_side'], evaluation)
gesture_pred = net.inference(data['scoremap'], data['hand_side'], evaluation)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
if (VARIANT == 'direct') or (VARIANT == 'bottleneck'):
    loss = tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_normed']))
elif VARIANT == 'local':
    loss += tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_local']))
elif VARIANT == 'local_w_xyz_loss':
    from utils.relative_trafo import bone_rel_trafo_inv

    # transform the local coordinates back into xyz for the loss
    coord3d_pred_xyz = bone_rel_trafo_inv(coord3d_pred)
    loss += tf.reduce_mean(tf.square(coord3d_pred_xyz - data['keypoint_xyz21_normed']))
elif VARIANT == 'proposed':
    loss += tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_can']))
    loss += tf.reduce_mean(tf.square(R - data['rot_mat']))
elif VARIANT == 'gesture':
    # Assume we can do this to whatever data['gesture'] is. However we decide to do this, if they represent which class
    # it is then it's fine
    loss += tf.reduce_mean(tf.square(int(gesture_pred == data['gesture'])))

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    _, loss_v = sess.run([train_op, loss])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
