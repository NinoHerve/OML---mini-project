# Pytorch - schedulers

#### Where ?
Schedulers available in `torch.optim.lr_scheduler` module.

#### How ?
Learning rate scheduling should be applied **after optimizer's update**:

Pseudo-code:
```python
scheduler = ...
for epoch in range(100):
	train(...)
	validate(...)
	scheduler.step()
```

Example:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # scheduler always takes as input at least the optimizer

for epoch in range(20):
	for input, target in dataset:
		optimizer.zero_grad()
		output = model(input)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()
	scheduler.step()   # update scheduler here
```

#### Who ? (13 candidates + 2 operators)
- [`LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR): Multiply initial `lr` by a given function.
	- `optimizer`
	- `lr_lambda`: Function taking `epoch` and returning a float.
	
- [`MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR): Multiply current `lr` by a given function.
	-> `optimizer`
	-> `lr_lambda`: Function taking `epoch` and returning a float.
	
- [`StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR): Multiply current `lr` by `gamma` every `step_size`.
	-> `optimizer`
	-> `step_size`
	-> `gamma`
	
- [`MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR): Multiply current `lr` by `gamma` at every milestone.
	-> `optimizer`
	-> `milestones`: List of epoch indices. Must be increasing.
	-> `gamma`

- [`ConstantLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR): Multiply initial `lr` by a constant factor until a specific number of epochs is reached.
	-> `optimizer`
	-> `factor`
	-> `total_iters`: Number of epochs during which to apply the scheduler.
	
- [`LinearLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR): Multiply initial `lr` by a factor that linearly changes until a specific number of epochs is reached.
	-> `optimizer`
	-> `start_factor`: Factor at first epoch.
	-> `end_factor`: Factor at last epoch.
	-> `total_iters`: Number of epochs during which to apply the scheduler.
	
- [`ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR): Multiply current `lr` by `gamma` at every epoch.
	-> `optimizer`
	-> `gamma`
	
- `PolynomialLR`:
	-> optimizer 
	
- [`CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR): Set `lr` using a cosine annealing schedule. Maximum of cosine is set to the initial `lr`.
	-> `optimizer`
	-> `T_max`: Maximum number of iterations. Once the maximum is reached, `lr` stays at `eta_min`. 
	-> `eta_min`: Minimum `lr`.
	
- [`CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts): Set `lr` using a cosine annealing schedule with warm restarts. Maximum of cosine is set to the initial `lr`.
	-> `optimizer`
	-> `T_0`: Number of iterations for the first restart.
	-> `T_mult`: Factor to increase/decrease $T_i$ at each restart.
	-> `eta_min`: Minimum `lr`.
	
- [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau): Multiply current `lr` by `factor` when a metric has stopped improving.
	-> `optimizer`
	-> `mode`: One of "min", "max". In "min" mode `lr` is reduced when metric stops decreasing. In "max" mode `lr` is reduced when metric stops increasing.
	-> `patience`: Number of epochs with no improvement after which `lr` will be reduced.
	-> `threshold`: 
	-> `threshold_mode`: One of "rel", "abs". In "rel" mode, `dynamic_threshold = best*(1+threshold)` in "max" mode or `best*(1-threshold)` in "min" mode. In "abs" mode, `dynamic_threshold = best+threshold` in "max" mode or `best-threshold` in "min" mode.
	-> `cooldown`: Number of epochs to wait after `lr` reduction.
	-> `min_lr`: Lower bound for `lr`.
	-> `eps`: Minimal decay applied to `lr`. If the difference between new and old `lr` is smaller than `eps`, the update is ignored.
	
- [`CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR): Set `lr` according to cyclical learning rate policy (CLR).
	-> `optimizer`
	-> `base_lr`: Lower bound for `lr`.
	-> `max_lr`: `lr` Upper bound for `lr`.
	-> `step_size_up`: Number of epochs in the increasing half of a cycle.
	-> `step_size_down`: Number of epochs in the decreasing half of a cycle.
	-> `mode`: One of "triangular", "triangular2", "exp_range". Type of window.
	-> `gamma`: Constant in "exp_range" scaling function.
	-> `scale_fn`: Function. Custom scaling policy defined by a single argument lambda function where $0 \le f(x) \le 1$.
	-> `scale_mode`: One of "cycle", "iterations". Defines whether `scale_fn` is evaluated on cycle number of cycle iterations. 
	-> `cycle_momentum`: If `True`, momentum is cycled inversely to learning rate between "base_momentum" and "max-momentum".
	-> `base_momentum`: 
	-> `max_momentum`:

- [`OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR):
	-> optimizer
	
- [`ChainedScheduler`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler): Chain schedulers.
	-> `schedulers`: List of schedulers.
	
- [`SequentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR): List of schedulers to be called sequentially. At each milestone the next scheduler is applied.
	-> `schedulers`: List of schedulers
	-> `milestones`: List of epoch indices.


#### References
1. https://pytorch.org/docs/stable/optim.html
