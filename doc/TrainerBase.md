```python
class TrainerBase:
  def __init__(self, model: ModelBase, user_predictor_cls,
               optimizer: typing.Union[Optimizer, None]):
    def train_one_batch(self, *args, **kwargs) -> dict:
      raise NotImplementedError()

  def get_training_data(self, rank: int, world_size: int):
    pass

  def train(self):
    def early_stop(self, batch_id, epoch_id, loss_history: list,
                   vali_error_history: list):
      return False
```