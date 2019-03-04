import os, glob, re, signal, sys, argparse, threading, time
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL import model
from PSNR import psnr
from TEST import test_VDSR

from dataset import DatasetFromHdf5
from random import shuffle

IMG_SIZE = (41, 41)
BATCH_SIZE = 32
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 120

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

if __name__ == '__main__':
    train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
    train_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

    train_output, weights = model(train_input)
    loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
    # for w in weights:
    #     loss += tf.nn.l2_loss(w)*1e-4
    tf.summary.scalar("loss", loss)

    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
    # tf.summary.scalar("learning rate", learning_rate)

    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # tf.train.MomentumOptimizer(learning_rate, 0.9)
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    opt = optimizer.minimize(loss, global_step=global_step)

    # saver = tf.train.Saver(weights, max_to_keep=0)

    # shuffle(train_list)
    config = tf.ConfigProto(
        # device_count={'GPU': 1}
    )

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()

        train_set = DatasetFromHdf5("data/train.h5")
        x = train_set.target.value.reshape((1000, IMG_SIZE[0], IMG_SIZE[0], 1))  # (1000,1,41,41) ndarray
        y = train_set.data.value.reshape((1000, IMG_SIZE[0], IMG_SIZE[0], 1))  # (1000,1,41,41) ndarray
        index = [i for i in range(1000)]

        print("\n===============START===============\n")
        for epoch in range(MAX_EPOCH):
            shuffle(index)
            train_set_x = x[index, :, :, :]
            train_set_y = y[index, :, :, :]
            for step in range(1000//BATCH_SIZE):
                offset = step*BATCH_SIZE
                input_data = train_set_x[offset:(offset+BATCH_SIZE), :, :, :]
                gt_data = train_set_y[offset:(offset+BATCH_SIZE), :, :, :]
                feed_dict = {train_input: input_data, train_gt: gt_data}
                _, curr_loss, output, g_step = sess.run([opt, loss, train_output, global_step], feed_dict=feed_dict)
                print("epoch %4d, iteration: %4d, loss %.4f\t lr %.5f" %
                      (epoch, step, np.sum(curr_loss)/BATCH_SIZE, learning_rate))
            saver.save(sess, "./checkpoints/VDSR_epoch_%03d.ckpt" % epoch)

