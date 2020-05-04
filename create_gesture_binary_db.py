"""
    Script to convert Rendered Handpose Dataset into binary files,
    which allows for much faster reading than plain image files.

    Set "path_to_db" and "set" accordingly.

    In order to use this file you need to download and unzip the dataset first.

    AFB -- rewriting this script to just use images and gesture labels
    Assumptions:
    SET is either 'training' or 'evaluation'
    PATH_TO_DB is the path to the gesture data set
    sample_id are the file names for the images (integers with leading zeros, see RHD)
    gesture classes are integers in {0 .. 5} (total of 6: forward, backward, left, right, stop, no-op)
    PATH_TO_DB/anno_SET.pickle is a pickled dictionary of the format {sample_id -> gesture class}
    Image files are located at PATH_TO_DB/SET/color/sample_id.png
    Image masks are located at PATH_TO_DB/SET/mask/sample_id.png
    Values of PATH_TO_DB and SET are set appropriately below in their variables (they're lower case below)
    NOTE: the annotation dictionary is a different format than before, which was {sample_id -> {k -> v}} with various keys
"""
from __future__ import print_function, unicode_literals
import numpy as np
import pickle
import os
import scipy.misc
import struct
from glob import glob

# SET THIS to where RHD is located on your machine
path_to_db = '.'
#good2go = glob("gesture_training/check/*")
#good2go = set(list(map(lambda x: x.split("/")[-1].split(".png")[0],good2go)))
# chose if you want to create a binary for training or evaluation set

set = 'gesture_training'
#set = 'gesture'

# ++++++++++++++++++++ No more changes below this line ++++++++++++++++++++


# function to write the binary file
def write_to_binary(file_handle, image, mask, gesture):
    """" Writes records to an open binary file. """
    bytes_written = 0
    # # 1. write kp_coord_xyz
    # for coord in kp_coord_xyz:
    #     file_handle.write(struct.pack('f', coord[0]))
    #     file_handle.write(struct.pack('f', coord[1]))
    #     file_handle.write(struct.pack('f', coord[2]))
    # bytes_written += 4*kp_coord_xyz.shape[0]*kp_coord_xyz.shape[1]

    # # 2. write kp_coord_uv
    # for coord in kp_coord_uv:
    #     file_handle.write(struct.pack('f', coord[0]))
    #     file_handle.write(struct.pack('f', coord[1]))
    # bytes_written += 4*kp_coord_uv.shape[0]*kp_coord_uv.shape[1]

    # # 3. write camera intrinsic matrix
    # for K_row in K_mat:
    #     for K_element in K_row:
    #         file_handle.write(struct.pack('f', K_element))
    # bytes_written += 4*9
    #
    # file_handle.write(struct.pack('B', 255))
    # file_handle.write(struct.pack('B', 255))
    # bytes_written += 2

    # 4. write image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            file_handle.write(struct.pack('B', image[x, y, 0]))
            file_handle.write(struct.pack('B', image[x, y, 1]))
            file_handle.write(struct.pack('B', image[x, y, 2]))
    bytes_written += 4*image.shape[0]*image.shape[1]*image.shape[2]

    # 5. write mask
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            file_handle.write(struct.pack('B', mask[x, y]))
    bytes_written += 4*mask.shape[0]*mask.shape[1]
    # write gesture class (integer)
    file_handle.write(struct.pack('B', gesture))

    # # 6. write visibility
    # for x in range(kp_visible.shape[0]):
    #     file_handle.write(struct.pack('B', kp_visible[x]))
    # bytes_written += kp_visible.shape[0]

    # print('bytes_written', bytes_written)


# binary file we will write
file_name_out = './data/bin/gesture_%s.bin' % set

if not os.path.exists('./data/bin'):
    os.mkdir('./data/bin')

# load annotations of this set
with open(os.path.join(path_to_db, set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

sample_ids = np.asarray(list(anno_all.keys()))
np.random.shuffle(sample_ids)
print(type(anno_all))
# iterate samples of the set and write to binary file
with open(file_name_out, 'wb') as fo:
    num_samples = len(anno_all.items())
    count_samples = 0
    for sample_id in sample_ids:
        gesture_class = anno_all[sample_id]
        # load data
        name = '%.5d.png' % sample_id
        if True:
        #if name[0:5] in good2go:
            image = scipy.misc.imread(os.path.join(path_to_db, set, 'color', name))
            mask = np.zeros((320, 320),dtype = "int32")
            # # get info from annotation dictionary
            # kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
            # kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
            # kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            # camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters
            # write_to_binary(fo, image, mask, kp_coord_xyz, kp_coord_uv, kp_visible, camera_intrinsic_matrix)
            write_to_binary(fo, image, mask, gesture_class)

        if (count_samples% 100) == 0:
            print('%d / %d images done: %.3f percent' % (count_samples, num_samples, count_samples*100.0/num_samples))
            print(gesture_class)
        count_samples+=1
