# Question Answer Generator Data Preparation
Question Answer generation is the task of generating pairs of questions and answers given some input text.
This library prepares the input data for training by converting it into pytorch datasets and splitting it into train and dev data.

The input data format for training has to be in SQUAD format. A single json file of the following format will be needed:
```
{
	"data": 
    [	
		{ 
		    "paragraphs": 
            [
				{ "context": "This is a test context. This is a good test.",
					"qas": 
                    [
						{ "question": "Is this a test?",
							"id": "1",
							"answers": 
                            [
								{ "answer_start": 1,
								  "text": "This is a test text"
								}
                                			        { "answer_start": 11,
								  "text": "test context"
								}
							]
						}
					]
		   	    }
			]
		}
	]
}
```

Thus for training you will be needing a list of paragraphs along with questions and answers generated from those paragraphs.


### Input
- `--data_source` Specify the location of the json file in the input cnvrg dataset.

  
### Output
The final output contains a `train_data_qg_hl_t5.pt` and `valid_data_qg_hl_t5.pt` files which are your pytorch datasets. These will be used for training. There will also be a `t5_qg_tokenizer` folder which contains the tokenizer used to create the training data. This will be used by the model for training as well.

### How to run

```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit_id for dataset}}]' --machine={compute name} --image={docker image name} --sync_before=false Train/prepare_data.py --data_source {path to input dataset}
```
Example run

```
cnvrg run  --datasets='[{id:"qna_trivia_data",commit:"5f1cd7c3fb68ae7c679f8c33966610670d32ff1e"}]' --machine="default.Large" --image=cnvrg:v5.0 --sync_before=false python3 Train/prepare_data.py --data_source /data/qna_trivia_data/traintriviafiltered.json

```

### Reference
https://github.com/patil-suraj/question_generation 
