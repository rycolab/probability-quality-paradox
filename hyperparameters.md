# gpt2_wp_medium

This model is a fine-tuned version of [gpt2-medium](https://huggingface.co/gpt2-medium) on the writingPrompts dataset.
It achieves the following results on the evaluation set:
- Loss: 2.9492

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 4.0336        | 0.09  | 250  | 3.0486          |
| 3.1036        | 0.19  | 500  | 3.0309          |
| 3.0845        | 0.28  | 750  | 3.0193          |
| 3.0745        | 0.38  | 1000 | 3.0118          |
| 3.0696        | 0.47  | 1250 | 3.0047          |
| 3.0557        | 0.56  | 1500 | 2.9997          |
| 3.0507        | 0.66  | 1750 | 2.9951          |
| 3.0481        | 0.75  | 2000 | 2.9909          |
| 3.0412        | 0.85  | 2250 | 2.9876          |
| 3.0388        | 0.94  | 2500 | 2.9848          |
| 3.0354        | 1.03  | 2750 | 2.9809          |
| 3.0072        | 1.13  | 3000 | 2.9784          |
| 3.002         | 1.22  | 3250 | 2.9771          |
| 3.0038        | 1.32  | 3500 | 2.9751          |
| 2.9964        | 1.41  | 3750 | 2.9721          |
| 2.9969        | 1.51  | 4000 | 2.9701          |
| 2.9961        | 1.6   | 4250 | 2.9680          |
| 2.9918        | 1.69  | 4500 | 2.9663          |
| 2.9919        | 1.79  | 4750 | 2.9635          |
| 2.9834        | 1.88  | 5000 | 2.9631          |
| 2.9863        | 1.98  | 5250 | 2.9605          |
| 2.9728        | 2.07  | 5500 | 2.9598          |
| 2.9569        | 2.16  | 5750 | 2.9588          |
| 2.9538        | 2.26  | 6000 | 2.9587          |
| 2.958         | 2.35  | 6250 | 2.9574          |
| 2.9549        | 2.45  | 6500 | 2.9566          |
| 2.9555        | 2.54  | 6750 | 2.9546          |
| 2.955         | 2.63  | 7000 | 2.9537          |
| 2.9482        | 2.73  | 7250 | 2.9531          |
| 2.9499        | 2.82  | 7500 | 2.9503          |
| 2.9476        | 2.92  | 7750 | 2.9504          |


### Framework versions

- Transformers 4.11.2
- Pytorch 1.7.1+cu110
- Datasets 1.12.1
- Tokenizers 0.10.1
