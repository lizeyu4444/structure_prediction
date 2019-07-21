# coding=utf-8

import os
import sys
import json
import copy
import logging
import functools

import numpy as np
import tensorflow as tf
# from tf_metrics import precision, recall, f1

from src.processer import NerProcessor
from src.models import Model
from src.bert import modeling
from src.bert import optimization

# tf.enable_eager_execution()

# Logging
# if not os.path.exists('results'):
#     os.makedirs('results')
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

 
def input_fn(filepath, params=None, shuffle_and_repeat=False):
    """Input function for estimator."""
    params = params if params is not None else {}

    dataset = np.load(filepath).item()
    for d in dataset:
        dataset[d] = np.array(dataset[d], dtype=np.int32)
    labels = dataset['label_ids']
    del dataset['label_ids']
    dataset = tf.data.Dataset.from_tensor_slices((dataset, labels))
    
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
        
    dataset =  dataset.batch(params.get('batch_size', 20)).prefetch(20)
    
    return dataset


def model_fn(features, labels, mode, params):
    """Model function.
        features: dict, each value is tensor, [N, S], except `nwords` is [N]
        labels: tensor, [N, S], label ids
    """ 
    nwords = features['nwords']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Init model
    crf_model = Model(params, is_training=is_training)
    logits, pred_ids, trans = crf_model.forwards(features)

    # Load weights
    tvars = tf.trainable_variables()
    if params.get('init_checkpoint'):
        assign_map, initialized_variables = \
            modeling.get_assignment_map_from_checkpoint(tvars, params.get('init_checkpoint'))
        tf.train.init_from_checkpoint(params.get('init_checkpoint'), assign_map)

    # Print loaded variables
    # for var in tvars:
    #     init_string = ""
    #     if var.name in initialized_variables:
    #         init_string = ", *INIT_FROM_CKPT*"
    #     print("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    # Estimator spec
    if mode == tf.estimator.ModeKeys.PREDICT:
        preds = {
            'pred_ids': pred_ids
        }
        export_outputs = {
            'predction': tf.estimator.export.PredictOutput(preds)
        }
        ##???
        # predictions is required when predicting
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=pred_ids,
                                          export_outputs=export_outputs)
    else:
        # Loss
        loss_likelihood, _ = tf.contrib.crf.crf_log_likelihood(inputs=logits, 
                                                               tag_indices=labels,
                                                               transition_params=trans,
                                                               sequence_lengths=nwords)
        loss = tf.reduce_mean(-loss_likelihood)

        # Metrics
        # weights = tf.sequence_mask(nwords)
        # metrics = {
        #     'acc': tf.metrics.accuracy(labels, pred_ids, weights),
        #     'precision': precision(labels, pred_ids, num_labels, indices, weights)
        #     'recall': recall(labels, pred_ids, num_labels, indices, weights)
        #     'f1': f1(labels, pred_ids, num_labels, indices, weights)
        # }

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, 
                                                     params['learning_rate'], 
                                                     params['num_train_steps'], 
                                                     params['num_warmup_steps'], 
                                                     False)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                "eval_loss": tf.metrics.mean_squared_error(labels, pred_ids)
            }
            # loss is required when eval
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def main(params_):
    # Copy
    params = copy.deepcopy(params_)

    # Modify params
    with open(os.path.join(params['output_dir'], params['tag_file']), 'r') as fi:
        tag_list = [t.strip() for t in fi.readlines() if t.strip()]
        params['num_labels'] = len(tag_list)

    if not os.path.exists(params['model_dir']):
        os.makedirs(params['model_dir'])

    # Input function
    train_file = os.path.join(params['output_dir'], params['train_file'])
    eval_file = os.path.join(params['output_dir'], params['eval_file'])

    num_examples = len(np.load(train_file).item()['input_ids'])
    params['num_train_steps'] = int(num_examples*1.0/params['batch_size']*params['epochs'])
    params['bert_config'] = modeling.BertConfig.from_json_file(params['bert_config'])

    train_inpf = functools.partial(input_fn, train_file, params=params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, eval_file)

    # Configs
    session_config = tf.ConfigProto(log_device_placement=False,
                                    inter_op_parallelism_threads=0,
                                    intra_op_parallelism_threads=0,
                                    allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(model_dir=params['model_dir'],
                                        save_checkpoints_steps=params['save_ckpt_steps'],
                                        keep_checkpoint_max=3,
                                        log_step_count_steps=params['log_every_steps'],
                                        session_config=session_config)

    # Estimator 
    estimator = tf.estimator.Estimator(model_fn, config=run_config, params=params)

    # Train and eval spec
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator=estimator,
                                                         metric_name='eval_loss',
                                                         max_steps_without_increase=200,
                                                         min_steps=5000,
                                                         run_every_secs=None,
                                                         run_every_steps=20)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf,
    #                                     max_steps=num_train_steps,
                                        hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf)

    # Train model
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Predict
    processer = NerProcessor(params['output_dir'])
    # id_2_label = processer.get_labels(reverse=True) # Not work now
    id_2_label = {i:l for l,i in label_2_id.items()}
    score_dir = 'results/score'
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    def write_predictions(name):
        with open(os.path.join(score_dir, '{}.preds.txt'.format(name)), 'w') as fi:
            test_file = os.path.join(params['output_dir'], '{}.npy'.format(name))
            test_inpf = functools.partial(input_fn, test_file)
            pred_ids = estimator.predict(test_inpf)
            test_examples = processer.load_examples(name)
            for example, ids in zip(test_examples, pred_ids):
                words, tags = example[0].split(), example[1].split()
                for word, tag, id in zip(words, tags, ids[1:]):
                    fi.write(' '.join([word, tag, id_2_label[id]]) + '\n')
                fi.write('\n')

    for name in ['train', 'eval', 'test']:
        write_predictions(name)


if __name__ == '__main__':

    print('x')
    param_file = 'src/params.json'
    with open(param_file, 'r') as fi:
        params = json.load(fi)

    main(params)

