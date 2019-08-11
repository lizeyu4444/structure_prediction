# coding=utf-8

import os
import time
import json
import glob

from tensorflow.contrib import predictor
from src.bert import tokenization
from src.processer import NerProcessor, map_fn_builder


# Load params
param_file = 'src/params_colab.json'
with open(param_file, 'r') as fi:
    params = json.load(fi)

processor = NerProcessor(params['output_dir'])
label_2_id = processor.get_labels()
# id_2_label = processor.get_labels(reverse=True)
id_2_label = {i:l for l,i in label_2_id.items()}
tokenizer = tokenization.FullTokenizer(vocab_file=params['vocab_file'], 
                                       do_lower_case=True)

# Load exported model
export_dir = sorted([d for d in glob.glob('./saved_model/*') if os.path.isdir(d)])[-1]
predict_fn = predictor.from_saved_model(export_dir)
map_fn = map_fn_builder(label_2_id, tokenizer, params['max_seq_length'])


def inference(data):
    # Main function
    if isinstance(data, str):
        data = (data, None)
    input_ids, input_mask, segment_ids, nwords = map_fn(data)
    
    st = time.time()
    predictions = predict_fn({
        'input_ids': [input_ids], 
        'input_mask': [input_mask],
        'segment_ids': [segment_ids],
        'nwords': [nwords]
    })
    print('Took {0:.3f}s to infer.'.format(time.time()-st))
    
    pred_ids = predictions['pred_ids'].tolist()[0]
    tags = list(map(lambda x:id_2_label[x], pred_ids))
    res = {
        'pred_ids': pred_ids[:nwords],
        'tags': tags[:nwords]
    }
    return res


if __name__ == '__main__':
    
    print('x')
    sent = '海钓比赛地点在厦门与进门之间的海域。'
    sent = tokenizer.tokenize(sent)
    inference(' '.join(sent))
