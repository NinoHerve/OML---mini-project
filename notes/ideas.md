
#### Schedulers by importance
There are 3 categories of schedulers.
- schedulers that decrease the learning rate over the epochs
- schedulers that decrease the learning rate when the validation score stagnates
- schedulter that oscillate the learning rate over the epochs

There are various implementations, leading to a plethora of schedulers. I suggest we focus our intention primarly on
one classic scheduler for each category. I will add some additional suggestions we can try if we want more results.

1. Main focus:
   - Exponential decay every N (~10) epochs with gamma = 0.1
   - Exponential decay on plateau with gamma = 0.1
   - Cyclic with triangular mode. Select parameters as in [paper]()
