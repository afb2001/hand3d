from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
from datetime import datetime
from nets.PosePriorNetwork import PosePriorNetwork
from nets.GestureCommandNetwork import GestureCommandNetwork
from data.BinaryDbReader import GestureDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot

# Chose which variant to evaluate
# VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
# VARIANT = 'proposed'
VARIANT = 'gesture'
USE_RETRAINED = True
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'
PATH_TO_GESTURE_SNAPSHOTS = "./snapshots_gesture/"

bs = 32


# training parameters
train_para = {'lr': [0.1, 0.001],
              'lr_iter': [60000],
              'max_iter': 20000,
              'show_loss_freq': 10,
              'snapshot_freq': 500,
              'snapshot_dir': 'snapshots_gesture'}

# get dataset # AFB - assume this is fine for now
dataset = GestureDbReader(mode='gesture_training',
                         batch_size=bs, shuffle=True,
                         crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

# build network graph
data = dataset.get()


# build network
# net = PosePriorNetwork(VARIANT)
net = GestureCommandNetwork()

# feed trough network
evaluation = tf.placeholder_with_default(True, shape=())
# _, coord3d_pred, R = net.inference(data['scoremap'], data['hand_side'], evaluation)
gesture_pred = net.inference(data['image'], evaluation, train=True)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
# Gesture class is the index of the max value in gesture_pred. Since data['gesture'] is an integer it's perfect.
# Loss is just number of incorrect predictions
# loss += tf.argmax(gesture_pred) != data['gesture']


#dg_op = tf.Print(data['gesture'], [data['gesture']])
#labels = tf.one_hot(dg_op, net.n_classes)

labels = tf.one_hot(data["gesture"], net.n_classes)


g = tf.reshape(tf.argmax(gesture_pred,axis = 1, output_type=tf.int32), (bs, 1))
accurate = tf.equal(g, tf.cast(data["gesture"], tf.int32))
accuracy = tf.count_nonzero(accurate)

with tf.control_dependencies([accuracy]):
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=gesture_pred))
#labels_op = tf.Print(labels, [labels], message = "hey look here")
#loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_op, logits=gesture_pred))

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)
sess.run(tf.global_variables_initializer())
# init weights
if USE_RETRAINED:
    last_cpt = tf.train.latest_checkpoint(PATH_TO_GESTURE_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    


saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1.0)
if USE_RETRAINED:
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    # retrained version: PoseNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    _, loss_v, acc = sess.run([train_op, loss, accuracy])

    if (i % train_para['show_loss_freq']) == 0:
        print(datetime.now())
        # print('Iteration %d\t Loss %.1e' % (i, loss_v))
        print('Iteration %d' % i)
        print(loss_v, acc, "/" + str(bs))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
