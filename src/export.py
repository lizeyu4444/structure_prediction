# coding=utf-8

import json
import tensorflow as tf

from train import model_fn


def serving_input_receiver_fn():
    """
    Serving input_fn that builds features from placeholders
    Returns:
      tf.estimator.export.ServingInputReceiver
    """
    input_ids = tf.placeholder(dtype=tf.string, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name='label_ids')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')

    features = {
        'input_ids': input_ids, 
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_ids': label_ids,
        'nwords': nwords
    }
    return tf.estimator.export.ServingInputReceiver(features, features)


if __name__ == '__main__':

    param_file = 'src/params.json'
    with open(param_file, 'r') as fi:
        params = json.load(fi)

    estimator = tf.estimator.Estimator(model_fn, 
                                       model_dir=params['model_dir'], 
                                       params=params)
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)




