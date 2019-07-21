# coding=utf-8

import os
import json
import copy
import tensorflow as tf

from src.train import model_fn
from src.bert import modeling


def serving_input_receiver_fn():
    """
    Serving input_fn that builds features from placeholders
    Returns:
      tf.estimator.export.ServingInputReceiver
    """
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')

    features = {
        'input_ids': input_ids, 
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'nwords': nwords
    }
    return tf.estimator.export.ServingInputReceiver(features, features)


def main(params_):
    # Copy
    params = copy.deepcopy(params_)

    # Load tag file
    with open(os.path.join(params['output_dir'], params['tag_file']), 'r') as fi:
        tag_list = [t.strip() for t in fi.readlines() if t.strip()]
        params['num_labels'] = len(tag_list)
    params['bert_config'] = modeling.BertConfig.from_json_file(params['bert_config'])

    checkpoint_path = tf.train.latest_checkpoint(params['model_dir'])
    print('Exporting {}...'.format(checkpoint_path.split('/')[-1]))

    estimator = tf.estimator.Estimator(model_fn, params=params)
    estimator.export_saved_model('saved_model', 
                                 serving_input_receiver_fn,
                                 checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    
    print('x')
    param_file = 'src/params.json'
    with open(param_file, 'r') as fi:
        params = json.load(fi)

    main(params)
