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
- ``lr_range_test.ipynb``: Performing the Learning Rate Range Test as explained [here](https://arxiv.org/pdf/1506.01186).
- ``lr_schedulers_test.ipynb``: Training different learning rate schedulers.
- ``lr_schedulers_visualizations.ipynb``: Visualizing different learning rate scheduling techniques.

Training parameters:
- ``parameters.yml``: YAML file with training parameters (e.g. ``n_epochs``, ``batch_size``, ``optimizer``, ``learning_rate_scheduler``, etc).

Results:
- ``tensorboard``: Directory containing folders for each set of experiments (tensorboard files).
- ``metrics``: Directory containing folders for each set of experiments (CSV files).


## Tensorboard
Tensorboard logs are stored in ``./src/tensorboard/`` store the training/evaluation metrics, more info [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html).

We suggest you inspect the tensorboard logs running the tensorboard interface using the following command:
```sh
tensorboard --logdir=./tensorboard/{folder_of_experiment}
```