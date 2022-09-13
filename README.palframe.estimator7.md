
# PAL Frame estimator7 (Up to 2022/9/13)

estimator7 is based on estimator5. Here, we give a brief introduction about `NEW FEATURES` in estimator7.


## New basic classes

   In estimator7, we introduce two extra base classes, i.e., `ModelBase` and `EvaluatorBase`. The responsibilities of `ModelBase` and `EvaluatorBase` list as follows:
   -  `ModelBase`: A wrapper of `nn.Module`, extends later to save model, load model state in `palframe` context
   - `EvaluatorBase`: Implemented a base class when do evaluation. In `EvaluatorBase`, two methods left to users to rewite, i.e., `evaluate_one_batch` and `metric`. Unlike estimator5, with `EvaluatorBase`, one can evaluate  one checkpoint and data independently. 

## Optimized  `TrainerBase`
  
  We first discard the `ModelWrapperBase` class, then the corresponding codes are transfered to `TrainerBase` class. Other update including:
  - New in `__init__` function: The required parameters include `param`, `model`, and optinal parameters include `optimizer`, `lr_scheduler`, `evaluator`.

  - New in `train` function: The  required parameter is `train_data`, and optional parameters include `dev_data`,  `test_data`. Inside the `train` function, `batch_id` replaced `smaple num` to participate in  train stop condition and learning rate decay. Using `batch_id`, `param.train_sample_num` is not necessary to set.


## New in `ParamBase`
  We reorder the hyperparameters declaraed in `__init__` function, which include six parts, i.e., `Train params`, `Optimal params`, `Draw Figure params`, `Distribution param`,  `Eval params`, `Pred params`. Meanwhile, function `distributed_init` is added in param module to initialize the distributed context. After call `distributed_init` function, one can use two functions `palframe.get_rank`, `palframe.get_world_size`  to get `rank` and `world size` respectively. Some new parameters in `__init__` function includes:

  - For evaluation: `param.metric`, `metric_primary_field`, `eval_value_is_large_better`, `eval_valid_file_extention`

  - For optimization: `num_warmup_steps`,`num_warmup_ratio`

  - For training: `train_valid_file_extention`, `max_train_step`, `eval_during_trainning`, `eval_gap_step_num`, `is_save_model_at_first_step`, `log_print_ranks`, `path_initial_model_load_optimizer`

   
## Unified launch command

  - Using estimator7, you can launch experiment with one gpu, multiple gpus but one node, mutiple gpus and multiple nodes in a unified command, i.e., 

      ```palframe start_dist_train --project_dir=PROJECT_DIR ```


  - estimator7 supports custom gpus on different nodes through speicifically setup on param `servers_file`. 
  
 -  There are also some arguments bind to `palframe start_dist_train` command, for examples,   `--servers_file`, `--train_files`, `--path_initial_model`, `--gpu_num` and so on.

## Attentions
  - Seting either `epoch_num` or  `max_train_step` can give a train stop condition.
  - You should set one of in parameters `eval_gap_step_num`, `eval_gap_sample_num`, `eval_gap_epoch_num` to `save` model, and set  bool parameter `eval_during_trainning` to do evaluation.
  

## Examples
  Coming soon...
    

