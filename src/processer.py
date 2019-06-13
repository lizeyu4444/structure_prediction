# coding=utf-8

import os
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
            self._read_data(os.path.join(data_dir, "example.train")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.dev")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.test")), "test")

    def save_examples(self, data, filename):
        self._write_data(data, os.path.join(self.output_dir, filename))

    def get_labels(self):
        if os.path.exists(os.path.join(self.output_dir, 'tags.txt')):
            with codecs.open(os.path.join(self.output_dir, 'tags.txt'), 'r') as f:
                label_list = [l.strip() for l in f.readlines() if l.strip()]
        else:
            self.labels = self.labels.union(set(['X', '[CLS]', '[SEP]']))
            label_list = list(self.labels)
            label_list.remove('O')
            label_list = ['O'] + label_list
            with codecs.open(os.path.join(self.output_dir, 'tags.txt'), 'w') as f:
                f.write('\n'.join(label_list))
        return dict(zip(label_list, range(len(label_list))))

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append((text.encode('utf-8'), label.encode('utf-8')))
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

    def _write_data(self, examples, output_file):
        """Writes a BIO data."""
        with codecs.open(output_file, 'w') as f:
            for example in examples:
                f.write('\t'.join(example) + '\n')



def map_fn_builder(label_2_id, tokenizer, max_seq_length):

    def map_fn(example):
        # tokenize
        tokens = tokenizer.tokenize(example[0])
        labels = example[1].split()
        if len(tokens) > max_seq_length-2:
            tokens = tokens[0:(max_seq_length-2)]
            labels = labels[0:(max_seq_length-2)]

        # for loop
        # input_ids: token ids
        # input_mask: mask to the input
        # segment_ids: distinguish the first sentence(0) and second one(1)
        # label_ids: label ids
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = ['[CLS]'] + labels + ['[SEP]']
        segment_ids = []
        label_ids = []
        for i, token in enumerate(tokens):
            segment_ids.append(0)
            label_ids.append(label_2_id[labels[i]])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        return input_ids, input_mask, segment_ids, label_ids
      
    return map_fn


if __name__ == '__main__':

    # Load params
    with open('bert_base/config.json', 'r') as fi:
        params = json.load(fi)

    init_checkpoint = params['init_checkpoint']
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    max_seq_length = params['max_seq_length']
    save_file = True

    processors = {
        "ner": NerProcessor
    }

    # Load data
    processor = processors['ner'](output_dir)
    train_examples = processor.get_train_examples(data_dir)
    eval_examples = processor.get_dev_examples(data_dir)
    test_examples = processor.get_test_examples(data_dir)

    # Save temp data
    # processor.save_examples(train_examples, 'train.txt')
    # processor.save_examples(eval_examples, 'eval.txt')
    # processor.save_examples(test_examples, 'test.txt')

    # Process data
    label_2_id = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(init_checkpoint, 'vocab.txt'), 
                                           do_lower_case=True)

    map_fn = map_fn_builder(label_2_id, tokenizer, max_seq_length)
    train_se = map(map_fn, train_examples)
    eval_se = map(map_fn, eval_examples)
    test_se = map(map_fn, test_examples)

    # Save processed data
    # [num_samples, 4, max_seq_length]
    if save_file:
        np.save(os.path.join(output_dir, params['train_file']), train_se)
        np.save(os.path.join(output_dir, params['eval_file']), eval_se)
        np.save(os.path.join(output_dir, params['test_file']), test_se)


