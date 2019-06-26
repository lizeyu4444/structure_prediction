# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
# from tf_metrics import accuracy, precision, recall, f1

from src.bert import modeling


class Model(object):

    def __init__(self, params, is_training=False):
        if params.get('bert_config'):
            self.bert_config = params['bert_config']
            del params['bert_config']
        else:
            raise ValueError('Must provide bert config!')
        self.params = params
        self.is_training = is_training

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


    def forwards(self, features):
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
        label_ids = features['label_ids']
        nwords = features['nwords']

        # Bert embeddings
        model = modeling.BertModel(config=self.bert_config,
                                   is_training=self.is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=self.params.get('use_one_hot_embeddings', False))
                                                                                # True for TPU else False
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
        trans = tf.get_variable('crf', [self.params['num_labels'], self.params['num_labels']])
        # preds_id: [N, S], best_score: [N]
        pred_ids, _ = crf.crf_decode(logits, trans, nwords)
        
        return logits, pred_ids, trans   

