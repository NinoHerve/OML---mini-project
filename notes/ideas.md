
#### Schedulers by importance
There are 4 categories of schedulers (to my opinion):
- schedulers that decrease the learning rate over the epochs
- schedulers that decrease the learning rate when the validation score stagnates
- schedulers that oscillate the learning rate over the epochs
- schedulers that oscillate the learning rate with decreasing amplitude over the epochs

There are various implementations, leading to a plethora of schedulers. I suggest we focus our intention primarly on
one classic scheduler for each category. I will add some additional suggestions we can try if we want more results.

1. Main focus:
   - Exponential decay with gamma = 0.95
   - Reduce on plateau with factor = 0.1
   - Cyclic policy with triangular mode. Select parameters as in [paper1](https://arxiv.org/pdf/1506.01186.pdf)
   - Cyclic policy with triangular2 mode. Select parameters as in [paper1](https://arxiv.org/pdf/1506.01186.pdf)
  
2. Interesting alternatives
   - Cosine annealing (type of oscillating scheduler)
   - Linear decay
   - One Cycle (cyclic scheduler that only has one cycle, so we start with low learning rate, it increases, and decreases again)
   - Step scheduler (decay at every step-size)
  
-
