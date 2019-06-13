params = {
	"task_name": "ner",
	"data_dir": "",
	"output_dir": "data/processed",
	"init_checkpoint": "",
	"train_file": "",
	"eval_file": "eval.npy",
	"test_file": "",
	"bert_config": "",
	"vocab_file": ""
}

import json


with open('results.json', 'w') as result_file:
    json.dump(params, result_file, indent=4)

