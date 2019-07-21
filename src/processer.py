# coding=utf-8

import os
import json
import codecs
import numpy as np

from src.bert import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):

    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.dev")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.test")), "test")

    def save_examples(self, examples, name):
        """Writes a BIO data."""
        with codecs.open(os.path.join(self.output_dir, name+'.proc'), 'w') as fi:
            for example in examples:
                fi.write('|||'.join(example) + '\n')

    def load_examples(self, name):
        """Loads a BIO data."""
        examples = []
        with codecs.open(os.path.join(self.output_dir, name+'.proc'), 'r') as fi:
            for line in fi.readlines():
                line = line.split('|||')
                if len(line) != 2: continue
                examples.append((line[0].strip(), line[1].strip()))
        return examples

    def get_labels(self, reverse=True):
        if os.path.exists(os.path.join(self.output_dir, 'tags.txt')):
            with codecs.open(os.path.join(self.output_dir, 'tags.txt'), 'r') as fi:
                label_list = [l.strip() for l in fi.readlines() if l.strip()]
        else:
            self.labels = self.labels.union(set(['X', '[CLS]', '[SEP]']))
            label_list = list(self.labels)
            label_list.remove('O')
            label_list = ['O'] + label_list
            with codecs.open(os.path.join(self.output_dir, 'tags.txt'), 'w') as fi:
                fi.write('\n'.join(label_list))
        if reverse:
            dict(zip(range(len(label_list)), label_list))
        return dict(zip(label_list, range(len(label_list))))

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append((text, label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                                self.labels.add(l)
                        lines.append([' '.join(word), ' '.join(label)])
                        words = []
                        labels = []
                # if contends.startswith("-DOCSTART-"):
                #     continue
            return lines


def map_fn_builder(label_2_id, tokenizer, max_seq_length):
    def map_fn(example):
        """Map function for an example.
        Args:
            tuple of (text, label), label can be None when inference
        return:
            input_ids: token ids
            input_mask: mask to the input
            segment_ids: distinguish the first sentence(0) and second one(1)
            label_ids: label ids
        """
        tokens = example[0].split() 
        if len(tokens) > max_seq_length-2:
            tokens = tokens[0:(max_seq_length-2)]
            
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        nwords = len(tokens)
        segment_ids = []
        for i, token in enumerate(tokens):
            segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # When inference, labels is None
        if example[1] is not None:
            labels = example[1].split()
            if len(labels) > max_seq_length-2:
                labels = labels[0:(max_seq_length-2)]
            labels = ['[CLS]'] + labels + ['[SEP]']
            label_ids = [label_2_id[label] for label in labels]
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
            return input_ids, input_mask, segment_ids, nwords, label_ids
        return input_ids, input_mask, segment_ids, nwords

    return map_fn


def main(params):
    # Params
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    max_seq_length = params['max_seq_length']

    # Load data
    processor = NerProcessor(output_dir)
    train_examples = processor.get_train_examples(data_dir)
    eval_examples = processor.get_dev_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)

    # Tokenize
    """
    Some dataset needs to be tokenized, but this dataset has been tokenized.
    """
    label_2_id = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=params['vocab_file'], 
                                           do_lower_case=True)

    # Save data
    processor.save_examples(train_examples, 'train')
    processor.save_examples(eval_examples, 'eval')
    processor.save_examples(test_examples, 'test')

    # Process and convert to model inputs
    map_fn = map_fn_builder(label_2_id, tokenizer, max_seq_length)
    train_se = list(map(map_fn, train_examples))
    eval_se = list(map(map_fn, eval_examples))
    test_se = list(map(map_fn, test_examples))
    
    # Change format
    train_se = {
        'input_ids': [i[0] for i in train_se], 
        'input_mask': [i[1] for i in train_se],
        'segment_ids': [i[2] for i in train_se],
        'nwords': [i[3] for i in train_se],
        'label_ids': [i[4] for i in train_se]
    }
    eval_se = {
        'input_ids': [i[0] for i in eval_se], 
        'input_mask': [i[1] for i in eval_se],
        'segment_ids': [i[2] for i in eval_se],
        'nwords': [i[3] for i in eval_se],
        'label_ids': [i[4] for i in eval_se]
    }
    test_se = {
        'input_ids': [i[0] for i in test_se], 
        'input_mask': [i[1] for i in test_se],
        'segment_ids': [i[2] for i in test_se],
        'nwords': [i[3] for i in test_se],
        'label_ids': [i[4] for i in test_se]
    }

    # Save input data
    np.save(os.path.join(output_dir, params['train_file']), train_se)
    np.save(os.path.join(output_dir, params['eval_file']), eval_se)
    np.save(os.path.join(output_dir, params['test_file']), test_se)


if __name__ == '__main__':

    param_file = 'src/params.json'
    with open(param_file, 'r') as fi:
        params = json.load(fi)

    main(params)

