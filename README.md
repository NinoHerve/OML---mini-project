# Learning Rate Scheduler Project
This project aims at exploring the impact of learning rate schedulers when training models. Specifically, we look at training from scratch an image NN architecture (MobileNetV3) ...

## Tasks
- Model: [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- Tasks (known image multiclass classification):
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    - torchvision.Dataset: [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10)
  - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
    - torchvision.Dataset: [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST)


## Files (structure)
Training:
- ``playground.ipynb``: 

Training parameters:
- ``parameters.yml``: YAML file with training parameters (e.g. ``n_epochs``, ``batch_size``, ``optimizer``, ``learning_rate_scheduler``, etc).

Exploration:
- ``lr_range_test.ipynb``: performing the Learning Rate Range Test.
- ``lr_schedulers.ipynb``: visualizing different learning rate scheduling techniques.


## Learning Rate Range Tests
When performing the [“LR range test”](https://arxiv.org/pdf/1506.01186.pdf) (i.e., running the model for several epochs while letting the learning rate increase linearly between low and high LR values), we identify reasonable minimum and maximum boundary values estimate for the learning rate (detailed in ``lr_range_test.ipynb``).


## Tensorboard

In this directory, the tensorboard logs should be stored when you're training/evaluating the models. Training and evaluation functions already include the code to push logs to tensorboard.

We suggest you inspect the tensorboard logs running the tensorboard interface using the following command:
```sh
tensorboard --logdir=./tensorboard
```

_You can find more information on Tensorboard with pytorch [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)._