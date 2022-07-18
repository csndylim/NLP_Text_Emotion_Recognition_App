# Text_Emotion_Recognition

Text Emotion Recognition Web App 

| Model | Training Acc | Training Loss | Validation Acc | Validation Loss | Runtime
| :---: | :---: | :---: | :---: | :---: | :---: |
| RoBERTa (base) | 89.942 | 0.3164 | 74.538 | 1.24 | 12h 57m 50s |
| BERT (base) | 38.449 | 1.789 | 39.002 | 1.854 | 10h 49min 33s |
| BERT (large) | 44.676 | 1.624| 43.67 | 1.794 | 11h 3m 24s |
| LSTM (64) | 40.18 | 2.054 | 0.2847 | 2.229 | 2h 11m 41s |
| LSTM (128) | 41.68 | 2.161 | 0.3018 | 2.30 | 2h 30m 50s |
| LSTM (256) | 42.20 | 2.123 | 0.3044 | 2.267 | 3h 22m 4s |
| CNN (64 Filters) | 39.09 | 1.881 | 0.2787 | 2.101 | 57m 49s |
| CNN (128 Filters) | 38.98 | 1.860 | 0.2770 | 2.1 | 58m 34s |
| CNN (256 Filters) | 44.84 | 1.728 | 0.307 | 2.046 | 59m 44s |

Fine-Tuned Model:
https://huggingface.co/junxtjx/roberta-base_TER/

RoBERTa_base:
https://wandb.ai/wasabee/Text_Emotion_Recognition_RoBERTa?workspace=user-wasabee

BERT_large:
https://wandb.ai/wasabee/Text_Emotion_Recognition_BERT_large_cased2?workspace=user-wasabee

BERT_base:
https://wandb.ai/wasabee/Text_Emotion_Recognition_BERT_base_cased2?workspace=user-wasabee

LSTM_64_Cells:
https://wandb.ai/wasabee/Text_Emotion_LSTM2_Cell_64?workspace=user-wasabee

LSTM_128_Cells:
https://wandb.ai/wasabee/Text_Emotion_LSTM2_Cell_128?workspace=user-wasabee

LSTM_256_Cells:
https://wandb.ai/wasabee/Text_Emotion_LSTM2_Cell_256?workspace=user-wasabee

CNN_64
https://wandb.ai/wasabee/Text_Emotion_CNN_64?workspace=user-wasabee

CNN_128
https://wandb.ai/wasabee/Text_Emotion_CNN_128?workspace=user-wasabee

CNN_256
https://wandb.ai/wasabee/Text_Emotion_CNN_256?workspace=user-wasabee


You can download the dataset from https://data.world/crowdflower/sentiment-analysis-in-text
