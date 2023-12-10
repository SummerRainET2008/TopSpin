# TrainerBase:

 1. Interface `__init__(self, model: ModelBase, user_predictor_cls, optimizer)`
    * `model`:  user-defined model, not encapusulated by DDP.
    * `user_predictor_cls`: class name of user-defined predictor.
    * `optimizer`: an optimizer instance or None. If set to None, TopSpin would generate an instance defined by `Param.optimizer_name`.

 1. Interface `train_one_batch(self, *args, **kwargs) -> dict`
    * Input: a batch tensor.
    * Return: a dictionary including loss, and possibly more auxiliary information, e.g., figure. 

 1. Interface `get_training_data(self, rank: int, world_size: int)`
    * Suppose there are `world_size` GPUs (or workers), and the current worker index is `rank`.
    * Then split your training data by `world_size` and return the `(rank + 1)-th` one.

 1. Interface `early_stop(self, batch_id, epoch_id, loss_history: list, vali_error_history: list):`
    * You can define your early stop codition. 
    * Or just `return False` to turn off early stop.
