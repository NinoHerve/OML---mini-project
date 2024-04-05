
## Schedulers by importance
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
  
## Building our own scheduler
We can use the scheduler `LambdaLR` or `MultiplicativeLR` to build our own schedulers. Both schedulers take as input a function. The latter must take as input the current number of epochs. At each step `LambdaR` will multiply the output of the function by the initial learning rate while `MultiplicativeLR` will multiply the output of the function by the current learning rate.

The generalized function for damped oscillators $$e^{-ax}[b_1\cos(c_1x) + b_2cos(c_2x)]$$ allows us to generate decreasing linear functions, decreasing exponentials, oscillations, damped oscillations englobing all the schedulers above. However, work is needed to range the parameters $a$, $b_1$, $b_2$, $c_1$, $c_2$ such that the generated schedulers don't go "out of hand".
