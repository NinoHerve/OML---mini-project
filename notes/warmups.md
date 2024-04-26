# Warmups in Deep Learning

#### Accurate, Largee Minibatch SGD, Training ImageNet in 1 Hour
June 2017
[link](https://arxiv.org/abs/1706.02677)

Introduction of warmups in Deep Learning in order to help train a model with large batch sizes.

#### A Closer Look at Deep Learning Heuristics, Learning Rate Restarts, Warmup and Distillation
October 2018
[link](https://arxiv.org/abs/1810.13243)

Compare warmup effects using VGG-16 architecture and CIFAR-10 dataset.
- 3 training sessions: (1) large batch + no warmup, (2) large batch + warmup, (3) small batch + no warmup.
- Large batch = 5000 and small batch = 100.
- Warmup using linear increase from 0 to 2.5 during <span style="color:yellow">200 iterations</span>. (if dataset=50 000 samples like use than 200 iterations are 20 epochs)
- Learning rate for small batch is 0.05 and for large it's 2.5 (scaling rule)

They use Canonical Correlation Analysis (CCA) to compare the weights of the model at
different iterations of the training. They conclude that "effect of learning rate warmup 
is to prevent the deeper layers from creating training instability".

#### Optimization for Deep Learning, An Overview
June 2020
[link](https://link.springer.com/article/10.1007/s40305-020-00309-6)

> **Learning Rate Warmup**: "Warmup" is a commonly used heuristic in deep learning. It means to use a very small learning rate for a number of iterations and then
increases to the "regular" learning rate. It has been used in a few major problems, including ResNet, large-batch training for image classification, and many popular 
natural language architectures such as Transformer networks BERT. See **A Closer Look at Deep Learning Heuristics, Learning Rate Restarts, Warmup and Distillation** 
for an empirical study of warmup.

> **Cyclical Learning Rate** An interesting variant is SGD with cyclical learning rate. The basic idea is to let the step-size bounce between a lower threshold and
an upper threshold. In one variant called SGDR (the paper of cosine annealing I have read before), the general principle is to gradually decrease and then gradually increase step-size within one epoch, and one special
rule is to use piecewise linear step-size. A later work reported "super convergence behavior" that SGDR converges several times faster than SGD in image classification.
In another variant, within one epoch the step-size gradually decreases to the lower threshold and suddenly increases to the upper threshold ("restart"). This "restart"
strategy resembles classical optimization tricks. Scientists studied the reasons of the success of cyclical learning rates, but a thorough understanding remains elusive.
