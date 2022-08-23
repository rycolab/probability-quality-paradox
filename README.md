# High probability or low information? The probability–quality paradox in language generation

## Human Evaluation
The human ratings can be found in `data/human_scores`

## Models & Datasets
We use the [Hugging Face](https://huggingface.co/) framework to train models and to generate prompts from the model instances.

### Abstractive Summarization
The large version of the BART can be loaded like this:
```python
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```
The [CNN/Daylimail](https://github.com/abisee/cnn-dailymail) dataset can also be accessed via the Hugging Face API:
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')
```

### Story Generation
#### Preprocessing
The preprocessing script can be found in `src/preproc_wp.py`. Download the [writingPrompts dataset](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz) and unzip the .tar into  the `data/datasets` folder. Then run `preproc_wp.py`.

The script creates for each of the train/test/valid splits a `<split>.comb.txt` file that contains one prompt-story pair per line.


#### Fine-Tuning
The finetuning script is located at `src/run_clm.py`. One can specify the model instance to be finetuned via the `--model_name_or_path` argument. We fine-tune an instance of `"gpt2-medium"`. The training and validation files can be passed in via the `--train_file` and `--validation_file` arguments. For a full overview of all available training args see the [Hugging Face Documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). A full list of the training hyperparamets can be found in `hyperparameters.md`.

The trained model instance can then be loaded into Hugging Face by passing the path to the saved model instance to the `from_pretrained` method.

### Unconditional Language Generation
#### Preprocessing
The preprocessing script can be found in `src/preproc_wiki.py`. Download the raw version of [WikiText 103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/#download) Dataset and unzip into  the `data/datasets` folder. Then run `preproc_wiki.py`.

The script creates for each of the train/test/valid splits a `wiki.<split>.processed.txt` file that contains one trainings example per line.


#### Fine-Tuning
Fine-tuning is done as for the Story Generation task.

## Forward Pass
To calculate the probability of the reference text under the model the scripts `src/news_forward.py`, `src/stories_forward.py` and `src/wiki_forward.py` are used.

## Generation
All texts in our work are generated using Hugging Face's `generate` method called on a model instance initialized via the `from_pretrained` method. All generation settings such as `num_beams`, `max_length`, `top_k`, `top_p` etc can be passed as parameters to the `generate` method. Note that most models come with default generation parameters. By passing no parameters, `generate` will fall back to the default parameters. Be sure to overwrite all parameters related to decoding to ensure comparability across models. For more details see the [Hugging Face Documentation](https://huggingface.co/docs/transformers/v4.14.1/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate).

#### MBR decoding
The MBR decoding framework requires to obtain multiple ancestral samples from the model. This can be achieved using the `generate` method and setting the `num_return_sequences` argument to the desired number. One can also include outputs of other decoding methods into the set of candidates. To then perform the actual minimum risk decodin∏g we use the following framework: https://github.com/Roxot/mbr-nmt
