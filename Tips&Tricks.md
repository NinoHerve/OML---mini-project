
## Learning range test
This test helps you find a suitable learning range when changing your architecture or dataset.

Run your model for several epochs while letting the learning rate increase linearly between low and high LR values. 
Next, plot the accuracy versus learning rate. Note the learning rate value when the accuracy starts to increase and 
when the accuracy slows, becomes ragged, or starts to fall. These two learning rates are good choices for bounds; 
that is, set `base_lr` to the first value and set `max_lr` to the latter value. Alternatively, one can use the rule 
of thumb that the optimum learning rate is usually within a factor of two of the largest one that converges and set 
`base_lr` to 1/3 or 1/4 of `max_lr`.

Example:
![rangeLR](https://github.com/NinoHerve/OML---mini-project/assets/117817842/75f5bff8-c4ab-4e2b-ad66-42433a7fdec2)

The Figure shows an example of making this type of run with the CIFAR-10 dataset, using the architecture and 
hyper-parameters provided by Caffe. One can see from the Figure that the model starts converging right away, 
so it is reasonable to set `base_lr`$= 0.001$. Furthermore, above a learning rate of $0.006$ the accuracy rise 
gets rough and eventually begins to drop so it is reasonable to set `max_lr`$=0.006$.
