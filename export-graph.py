from __future__ import print_function

import argparse
import cv2
import numpy as np
from src import transform
import tensorflow as tf
import os


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str,
                    help='checkpoint in dir', default='models/kanagawa')
parser.add_argument('--out-dir', type=str,
                    help='dir to save checkpoint out', default='ncs_graph')


def main():
    options = parser.parse_args()

    with tf.Graph().as_default():
        img_shape = (224, 224, 3)
        
        batch_shape = (1,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='input')
        preds = transform.net(img_placeholder)
        output = tf.identity(preds, name='output')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())

            if os.path.isdir(options.checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(options.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                saver.restore(sess, options.checkpoint_dir)

            out_path = os.path.join(options.out_dir, 'ncs')
            saver.save(sess, out_path)

            print("Saved to:", out_path + '.meta')
            print('Input tensor name:', img_placeholder.name)
            print('Output tensor name:', output.name)


if __name__ == '__main__':
    main()
