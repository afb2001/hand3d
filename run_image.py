import tensorflow as tf
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

# training parameters
train_para = {'lr': [1e-5, 1e-6],
              'lr_iter': [60000],
              'max_iter': 80000,
              'show_loss_freq': 10,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_gesture'}

# get dataset # AFB - assume this is fine for now
#dataset = GestureDbReader(mode='gesture_training',
#                         batch_size=8, shuffle=True,
#                         crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

# build network graph
#data = dataset.get()
# build network
# net = PosePriorNetwork(VARIANT)
net = GestureCommandNetwork()
dataset = GestureDbReader(mode='training',
                         batch_size=8, shuffle=True,
                         crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)
data = dataset.get()

# feed trough network
evaluation = tf.placeholder_with_default(True, shape=())
# _, coord3d_pred, R = net.inference(data['scoremap'], data['hand_side'], evaluation)
#inference = net.inference(data["image"], evaluation, train=False)

inference = net.infer_gesture_as_int(data["image"], evaluation, train=False)
op = tf.Print(data["gesture"], [data["gesture"]])
# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)
if USE_RETRAINED:
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    # retrained version: PoseNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    last_cpt = tf.train.latest_checkpoint(PATH_TO_GESTURE_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

sess.run(tf.global_variables_initializer())



for i in range(16):
    x, y = sess.run([inference, data["gesture"]])
    print(x, y.T)
