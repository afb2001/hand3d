from __future__ import print_function, unicode_literals

import tensorflow as tf
import os

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import *
from utils.canonical_trafo import *
from utils.relative_trafo import *

ops = NetworkOps


class GestureCommandNetwork(object):
    """ Network containing different variants for lifting 2D predictions into 3D. """
    def __init__(self):
        self.num_kp = 21
        self.n_fully_connected_layers = 2
        self.n_classes = 6
        self.fully_connected_layers_size = 512
        self.color_hand_pose_net = ColorHandPose3DNetwork()

    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        # I think the weight files should initialize all weights, including ones for this network
        # self.color_hand_pose_net.init(session, weight_files, exclude_var_list)

        import pickle
        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def inference(self, image, evaluation, train=False):
        keypoints_scoremap, image_crop, scale_crop, center = self.color_hand_pose_net.inference2d(image)
        return self.inference_gesture(keypoints_scoremap, evaluation, train)

    def inference_gesture(self, keypoints_scoremap, evaluation, train=False):
        """ Inference of command class for vehicle remote driving from gestures. """
        keypoints_scoremap = tf.nn.avg_pool(keypoints_scoremap, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
        # assume one hand side
        # hand_side = 0  # ??? TODO! -- do we need to train on hand_side? should always be the same
        with tf.variable_scope('Gesture'):
            x = keypoints_scoremap  # this is 28x28x21
            s = x.get_shape().as_list()

            # Add two fully connected layers
            out_chan_list = [self.fully_connected_layers_size for _ in range(self.n_fully_connected_layers)]
            x = tf.reshape(x, [s[0], -1])
            # x = tf.concat([x, hand_side], 1)
            for i, out_chan in enumerate(out_chan_list):
                x = ops.fully_connected(x, 'fc_%d' % i, out_chan=out_chan, trainable=train)
                x = ops.dropout(x, 0.8, evaluation)

            # infer gesture class
            gesture_class = ops.fully_connected(x, 'fc_gesture', out_chan=self.n_classes, trainable=train)

            # softmax layer at the end to make it look like probabilities (not sure it's necessary,
            # might be bad depending on loss function)
            # gesture_class = tf.nn.softmax(gesture_class)

            return gesture_class
