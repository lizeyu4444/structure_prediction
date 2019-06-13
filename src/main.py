# coding=utf-8

import os
import sys
import json
import logging
import functools

import numpy as np
import tensorflow as tf
# from tf_metrics import precision, recall, f1
from src.models import model_fn

tf.enable_eager_execution()

DATADIR = 'data/processed'

# Logging
if not os.path.exists('results'):
    os.makedirs('results')
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def input_fn(filepath, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}

    dataset = np.load(filepath)
    # features = {
    #     'input_ids': dataset[:, 0, :],
    #     'input_mask': dataset[:, 1, :],
    #     'segment_ids': dataset[:, 2, :],
    #     'label_ids': dataset[:, 3, :]
    # }

    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset =  dataset.batch(params.get('batch_size', 20)).prefetch(20)
    return dataset


if __name__ == '__main__':

    # Params
    with open('bert_base/params.json', 'r') as fi:
        params = json.load(fi)

    eval_file = os.path.join(params['output_dir'], params['eval_file'])
    
    with open(os.path.join(params['output_dir'], params['tag_file']), 'r') as fi:
    	tag_list = [t.strip() for t in fi.readlines() if t.strip()]
    	params['num_labels'] = len(tag_list)

    with open(os.path.join(params['init_checkpoint'], 'bert_config.json'), 'r') as fi:
    	bert_config = json.load(fi)
    	params['bert_config'] = bert_config


    # Input function
    train_inpf = functools.partial(input_fn, train_file, params=params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, eval_file)

    # Configs
    session_config = tf.ConfigProto(log_device_placement=False,
                                    inter_op_parallelism_threads=0,
                                    intra_op_parallelism_threads=0,
                                    allow_soft_placement=True)
    session_config.gpu_options.allow_groth = True
    run_config = tf.RunConfig(model_dir=params['model_dir'],
                              save_checkpoints_steps=500,
                              session_config=session_config)

    # Estimator 
    estimator = tf.estimator.Estimator(model_fn, config=run_config, params=params)

    # Train and eval spec
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator=estimator,
                                                         metric_name='eval_loss',
                                                         max_steps_without_increase=500,
                                                         min_steps=5000,
                                                         run_every_steps=params['save_chpt_steps'])
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf,
                                        max_steps=num_train_steps, ##?
                                        hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpfn)

    # Train model
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        if not os.path.exists('results/score'):
             os.makedirs('results/score')
        with open('results/score/{}.preds.txt'.format(name), 'wb') as f:
            test_inpf = functools.partial(input_fn, '{}.npy'.format(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(' '.join([word, tag, tag_pred]) + '\n')
                f.write('\n')

    # for name in ['train', 'eval', 'test']:
    #     write_predictions(name)

