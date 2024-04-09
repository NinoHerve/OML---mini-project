# Learning Rate Scheduler Project
This project aims at exploring the impact of learning rate schedulers when training models. Specifically, we look at fine-tunning a Language Model encoder in four different tasks.

## Tasks
- Model: BERT style architecture (i.e., RoBERTa)
- Tasks (taken from the different fine-tunning task specified for BERT in the [original paper](https://arxiv.org/pdf/1810.04805.pdf)):
  - Single Sentence Classification Task (Sentiment Analysis)
    - Dataset: [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)
  - Sentence Pair Classification Task (Natural Language Inference, aka textual entailment)
    - Dataset: [MNLI](https://cims.nyu.edu/~sbowman/multinli/) 
    - Paper: https://arxiv.org/pdf/1704.05426.pdf
  - Single Sentence Tagging Task (Name Entity Recognition)
    - Dataset: [CoNLL-2003 NER](https://paperswithcode.com/dataset/conll-2003)
  - Question Answering Task
    - Dataset: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)


## Tensorboard

In this directory, the tensorboard logs should be stored when you're training/evaluating the models. Training and evaluation functions already include the code to push logs to tensorboard.

We suggest you inspect the tensorboard logs running the tensorboard interface using the following command:
```sh
tensorboard --logdir=./tensorboard
```

_You can find more information on Tensorboard with pytorch [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)._