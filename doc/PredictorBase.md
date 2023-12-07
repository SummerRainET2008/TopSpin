```python

class PredictorBase:
  def __init__(self, model: ModelBase):

  def load_model(self):

  def predict(self, *batch):

  def evaluate_file(self, data_file) -> float:
    raise NotImplementedError()

```