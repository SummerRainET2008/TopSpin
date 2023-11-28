
# PAL Frame Commands

We support some commands to create new project, launch training , stop training ...

```
usage: palframe [-h] {start_dist_train,stop_dist_train} ...

palframe command tools

positional arguments:
  {start_dist_train,stop_dist_train}
                        sub-command
    init                init a new project
    start_dist_train    start a dsitributed trainning
    stop_dist_train     stop a dsitributed trainning

optional arguments:
  -h, --help            show this help message and exit
```
- create new project 

  Execute following command:

  ```palframe init``` 
  
  then give some answer to palframe , such as name,email and so on.

- Start distributed trainning
```shell
  usage: palframe start_dist_train [-h] [--debug_level DEBUG_LEVEL]
                                 [--train_script_name TRAIN_SCRIPT_NAME]
                                 [--trainer_cls_name TRAINER_CLS_NAME]
                                 [--param_script_name PARAM_SCRIPT_NAME]
                                 [--param_cls_name PARAM_CLS_NAME]
                                 --project_dir PROJECT_DIR
                                 [--extra_run_tag EXTRA_RUN_TAG]
                                 [--servers_file SERVERS_FILE]
                                 [--train_files TRAIN_FILES]
                                 [--vali_files VALI_FILES]
                                 [--test_files TEST_FILES]
                                 [--path_initial_model PATH_INITIAL_MODEL]
                                 [--gpus GPUS] [--epoch_num EPOCH_NUM]
                                 [--max_train_step MAX_TRAIN_STEP]
                                 [--model_saved_num MODEL_SAVED_NUM]
                                 [--iter_num_update_optimizer ITER_NUM_UPDATE_OPTIMIZER]
                                 [--batch_size BATCH_SIZE]

  optional arguments:
    -h, --help            show this help message and exit
    --debug_level DEBUG_LEVEL
                          logger level
    --train_script_name TRAIN_SCRIPT_NAME
                          name of train.py
    --trainer_cls_name TRAINER_CLS_NAME
                          class name of trainer
    --param_script_name PARAM_SCRIPT_NAME
                          name of param.py
    --param_cls_name PARAM_CLS_NAME
                          class name of param
    --project_dir PROJECT_DIR
                          location of project path
    --extra_run_tag EXTRA_RUN_TAG
                          extra run tag
    --servers_file SERVERS_FILE
                          servers file,e.g. ip1,ip2,ip3
    --train_files TRAIN_FILES
                          train files location
    --vali_files VALI_FILES
                          validation files location
    --test_files TEST_FILES
                          test files location
    --path_initial_model PATH_INITIAL_MODEL
                          inital checkpoint
    --gpus GPUS           gpus, such as `[0,1,2]`
    --epoch_num EPOCH_NUM
                          train stop condition: total epoch
    --max_train_step MAX_TRAIN_STEP
                          train stop condition:: max train step
    --model_saved_num MODEL_SAVED_NUM
                          max checkpoint num to save
    --iter_num_update_optimizer ITER_NUM_UPDATE_OPTIMIZER
                          gradient accumulation num
    --batch_size BATCH_SIZE
                          batch size
    
  ```
- Stop distributed trainning

  ```shell

  usage: palframe stop_dist_train [-h] [--debug_level DEBUG_LEVEL]
                                [--path_work PATH_WORK] [--servers SERVERS]
                                [--servers_file SERVERS_FILE]

  optional arguments:
    -h, --help            show this help message and exit
    --debug_level DEBUG_LEVEL
                          logger level
    --path_work PATH_WORK
    --servers SERVERS     ip1,ip2,ip3
    --servers_file SERVERS_FILE
```