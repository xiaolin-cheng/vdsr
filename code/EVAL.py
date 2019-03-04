import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
from PSNR import psnr
import scipy.io
import pickle
from MODEL import model
#from MODEL_FACTORIZED import model_factorized
import time
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

DATA_PATH = "./data/Set5_mat/*"

if __name__ == '__main__':
    train_input = tf.placeholder(tf.float32, shape=(1, None, None, 1))
    train_output, weights = model(train_input)

    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    saver = tf.train.Saver()
    model_ckpt = "./checkpoints/VDSR_epoch_020.ckpt"
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_ckpt)
        print("Testing model", model_ckpt, "\n")

        scales = [2, 3, 4]
        image_list = glob.glob(DATA_PATH)

        for scale in scales:
            avg_psnr_predicted = 0.0
            avg_psnr_bicubic = 0.0
            avg_elapsed_time = 0.0
            count = 0.0
            for image_name in image_list:
                # print("image name: ", image_name)
                if str(scale) in image_name:
                    print("Processing ", image_name)
                    # keys: 'im_gt_y', 'im_b_y', 'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l'
                    im_gt_y = sio.loadmat(image_name)['im_gt_y']
                    im_b_y = sio.loadmat(image_name)['im_b_y']

                    im_gt_y = im_gt_y.astype(float)/255.
                    im_b_y = im_b_y.astype(float)/255.

                    psnr_bicubic = psnr(im_gt_y, im_b_y, scale)
                    avg_psnr_bicubic += psnr_bicubic

                    input_array = np.resize(im_b_y, (1, im_b_y.shape[0], im_b_y.shape[1], 1))
                    start_t = time.time()
                    img_vdsr = sess.run(train_output, feed_dict={train_input: input_array})
                    end_t = time.time()
                    img_vdsr = np.resize(img_vdsr, (im_b_y.shape[0], im_b_y.shape[1]))

                    elapsed_time = end_t - start_t
                    avg_elapsed_time += elapsed_time

                    psnr_predicted = psnr(im_gt_y, img_vdsr, scale)
                    avg_psnr_predicted += psnr_predicted

                    count += 1

            print("Scale=", scale)
            print("PSNR_predicted=", avg_psnr_predicted / count)
            print("PSNR_bicubic=", avg_psnr_bicubic / count)
            print("It takes average {}s for processing".format(avg_elapsed_time / count), "\n")
