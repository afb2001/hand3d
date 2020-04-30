from __future__ import print_function, unicode_literals

import tensorflow as tf
import os

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

    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

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

    def inference(self, scoremap, hand_side, evaluation):
        """ Infer gesture from 2D scoremaps. """
        scoremap_pooled = tf.nn.avg_pool(scoremap, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        gesture_inference = self.inference_gesture(scoremap_pooled, evaluation, train=True)

        return gesture_inference

    def inference_gesture(self, keypoints_scoremap, evaluation, train=False):
        """ Inference of command class for vehicle remote driving from gestures. """
        # assume one hand side
        hand_side = 0  # ??? TODO! -- find out what data type hand_side has
        with tf.variable_scope('Gesture'):
            x = keypoints_scoremap  # this is 28x28x21
            s = x.get_shape().as_list()

            # Add two fully connected layers
            out_chan_list = [self.fully_connected_layers_size for _ in range(self.n_fully_connected_layers)]
            x = tf.reshape(x, [s[0], -1])
            x = tf.concat([x, hand_side], 1)
            for i, out_chan in enumerate(out_chan_list):
                x = ops.fully_connected(x, 'fc_%d' % i, out_chan=out_chan, trainable=train)
                x = ops.dropout(x, 0.8, evaluation)

            # infer gesture class
            gesture_class = ops.fully_connected(x, 'fc_gesture', out_chan=self.n_classes, trainable=train)

            # softmax layer at the end to make it look like probabilities (not sure it's necessary)
            gesture_class = tf.nn.softmax(gesture_class)

            return gesture_class
