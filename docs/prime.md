
### 1. Enable SPMD
Globally (outside of main):
```python
import torch_xla.runtime as xr
xr.use_spmd()
assert xr.is_spmd() is True
```

### 2. Initialize Model
Within main:
```python
@contextmanager
def set_default_dtype(dtype):
  # Get the current default dtype
  previous_dtype = torch.get_default_dtype()
  # Set the new default dtype
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    # Revert to the original default dtype
    torch.set_default_dtype(previous_dtype)

# Set the model dtype to bfloat16, and set the default device to the XLA device.
# This will capture the model constructor into a graph so that we can add
# sharding annotations to the weights later, and run the constructor on the XLA device.
with set_default_dtype(model_dtype), torch_xla.device():
    model = initialize_model_class(config.model)
```

### 3. Set Up Sharding
```python
# Sharding setup
model, self.input_sharding_spec, self.minibatch = setup_sharding_and_mesh(
    model, config
)
```

### 4. Set Up Transformations
```python
model = add_activation_checkpointing_and_scan(model, config)
model = add_optimization_barriers(model, config)
```

### 5. Create Optimizer
```python
self.optimizer = Adafactor(
      params=model.parameters(),
      lr=self.config.optimizer.learning_rate,
      relative_step=False,
      scale_parameter=False,
    )
self.lr_scheduler = get_scheduler(optimizer=optimizer)
```

### 6. Execute Initialization
```python
torch_xla.sync()
```

### 7. Prepare Model
```python
self.model.train()
self.model.zero_grad()
```

### 8. Train Step
```python
@torch_xla.compile(full_graph=True)
def train_step(self, batch: dict):
    loss, aux = self.model(**batch)
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return aux
```
Within training loop:
```python
aux = train_step(batch)
```

### 9. Step Closure
After train_step():
```python
xm.add_step_closure(
    step_closure,
    args=(epoch, step, aux),
    run_async=True,
)
```