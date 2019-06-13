# coding=utf-8

import tensorflow as tf
# from tf_metrics import accuracy, precision, recall, f1

from src.bert import modeling
# from src.layers import LSTM_CRF


class CrfModel(object):

    def __init__(self, params, bert_config=None):
        self.params = params
        self.bert_config = bert_config if bert_config else {}

    def _witch_cell(self):
        """
        RNN type
        return: 
            cell: RNN cell
        """
        cell = None
        if self.params['cell_type'] == 'lstm':
            cell = rnn.LSTMCell(self.params['hidden_size'])
        elif self.params['cell_type'] == 'gru':
            cell = rnn.GRUCell(self.params['hidden_size'])
        elif self.params['cell_type'] == 'lstm_block':
            cell = rnn.LSTMBlockFusedCell(self.params['hidden_size'])
        else:
            raise ValueError('Unknown cell type.')
        return cell


    def _bidirectional_rnn(self):
        """
           Bidirectional RNN
        return:
            cell_fw: forward_cell
            cell_bw: backward_cell
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.is_training:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.params['dropout'])
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.params['dropout'])
        return cell_fw, cell_bw


    def forwards(self, features, labels):
        '''
        Forward input to output
        Args:
            features: dict, input features
            labels: tensor([N, S]), batch of sentence labels
        return:
            logits: final layer before crf
            pred_ids: predicted word ids of crf layer
            trans: transition matrix
        '''
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']

        # Bert embeddings
        model = modeling.BertModel(config=self.bert_config,
                                   is_training=self.is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=use_one_hot_embeddings) # TPU:True other:False
        # [N, S, D]
        embeddings = model.get_sequence_output()
        embeddings = tf.layers.dropout(embeddings, 
                                       rate=self.params['dropout'], 
                                       training=self.is_training)

        # RNN layers
        if self.params['num_layers'] > 0:
            cell_fw, cell_bw = self._bidirectional_rnn()
            if self.params['num_layers'] > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw]*self.params['num_layers'], state_is_tupe=True)
                cell_bw = rnn.MultiRNNCell([cell_bw]*self.params['num_layers'], state_is_tupe=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embeddings,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = embeddings

        # Project final layer
        # [N, S, T]
        outputs = tf.layers.dropout(outputs, 
                                    rate=self.params['dropout'], 
                                    training=self.is_training)
        logits = tf.layers.dense(outputs, self.params['num_labels'])

        # CRF 
        # [T, T]
        trans = tf.get_varible('crf', [self.params['num_labels'], self.params['num_labels']])
        # preds_id: [N, S], best_score: [N]
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, trans, nwords)
        
        return logits, pred_ids, trans   


def model_fn(features, labels, mode, params):
    '''
    Model function, only features and labels are required, others are optional.
    params features: ([N, S], [N]), first item returned from input_fn
    params labels: [N, S], second item returned from input_fn
    '''
    bert_config = params['bert_config']
    del params['bert_config']

    # Init model
    crf_model = CrfModel(params, bert_config)
    logits, preds_id, trans = crf_model.forwards(features, labels)

    # Load weights
    tvars = tf.trainable_variables()
    if init_checkpoint:
        assign_map, initialized_variables = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                         init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assign_map)

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
        # predictions is required when predicting
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predctions=predctions,
                                          export_outputs=export_outputs)
    else:
        # Loss
        loss_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, trans, 
                                                               params['max_seq_length'])
        loss = tf.reduce_mean(-loss_likelihood)

        # Metrics
        # weights = tf.sequence_mask(nwords)
        # metrics = {
        #     'acc': tf.metrics.accuracy(labels, pred_ids, weights),
        #     'precision': precision(labels, pred_ids, num_labels, indices, weights)
        #     'recall': recall(labels, pred_ids, num_labels, indices, weights)
        #     'f1': f1(labels, pred_ids, num_labels, indices, weights)
        # }
        metrics = {
            "eval_loss": tf.metrics.mean_squared_error(labels, pred_ids)
        }

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, 
                                                     params['learning_rate'], 
                                                     # params['num_train_steps'], 
                                                     params['num_warmup_steps'], 
                                                     False)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # loss is required when eval
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


