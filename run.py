#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from datetime import datetime as time

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, load_weights_from_snapshot

if __name__ == '__main__':
    # images to be shown
    image_list = [
        "./results/devin2.png"
    ]
    #image_list.append('./data/img.png')
    #image_list.append('./data/img2.png')
    #image_list.append('./data/img3.png')
    #image_list.append('./data/img4.png')
    #image_list.append('./data/img5.png')
    image_list = list(glob("./gesture_training/color/*.png"))[::10]
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[0.0, 1.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)
    USE_RETRAINED = True
    PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true
    PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #c = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session(config=c)
    # initialize network
    net.init(sess)
    if USE_RETRAINED:
        # retrained version: HandSegNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

        # retrained version: PoseNet
        last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
        assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    # Feed image list through network
    for i in range(len(image_list)):
        img_name = image_list[i]
        start = time.now()
        image_raw = scipy.misc.imread(img_name)
        image_raw = scipy.misc.imresize(image_raw, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        hand_scoremap_v, image_crop_v, scale_v, center_v,\
        keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                             keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                            feed_dict={image_tf: image_v})

        hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
        print(time.now() - start)

        # visualize
        fig = plt.figure(1)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.imshow(image_raw)
        plot_hand(coord_hw, ax1)
        ax2.imshow(image_crop_v)
        plot_hand(coord_hw_crop, ax2)
        ax3.imshow(np.argmax(hand_scoremap_v, 2))
        plot_hand_3d(keypoint_coord3d_v, ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 1])
        ax4.set_zlim([-3, 3])
        plt.savefig("gesture_training/check/" + img_name.split("/")[-1] + "_t1_out.png")
        plt.close()
