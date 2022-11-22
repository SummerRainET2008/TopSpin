#coding: utf8
#author: zhouxuan553
# default params used in ParamBase

DEFAULT_PARAMS = dict(
    #### Train params  ####
    train_files=None,
    train_valid_file_extension=["pkl", "pydict", "json"],
    train_batch_size=32,
    train_sample_num=None,
    iter_num_update_optimizer=1,
    epoch_num=None,  # can be float.
    max_train_step=None,  #
    find_unused_parameters=True,
    # whether save first step
    is_save_model_at_first_step=True,
    model_saved_num=3,  # model save param
    model_save_stratyge='auto', #  `recent`: save recent, `top-k`: save best top-k, `auto` 
    # worker num in dataloader
    train_num_workers_loading_data=2,
    # worker num in processing example, i.e. create feat stage
    train_process_example_num_worker=1,
    # For the case that each GPU worker consumes different batch size.
    true_gradient=False,
    # ranks to print log, default is [0]
    log_print_ranks=[0],
    # Mixed-precision optimization
    use_amp=True,
    seed=0,
    # parallel num on one node, attention that
    gpu_num=1,
    use_gpu=False,
    # visible gpus
    gpus=None,
    # model for train
    train_path_initial_model=None,
    train_path_initial_model_load_optimizer=False,
    # restore from a path work, while load more information from last train
    path_work_restored_training=None,
    #### Optimal params ####
    # If optimizer is set as SGD, then lr decay is forced to the classic
    # step-wise decay, and warmup_ratio, ending_lr_ratio becomes void.
    optimizer_name="AdamW",
    lr=0.001,
    #  Such in RoBERTa, l2 weight decay is 0.01
    weight_decay=0.01,
    # "linear" or "cosine" or "cosine_with_restarts"  or "polynomial"
    # or "constant" or "constant_with_warmup"
    lr_scheduler_type='linear',  
    num_warmup_steps=None,
    num_warmup_ratio=None,
    param_clip_norm=1,
    #### Draw Figure params  ####
    # draw the loss figure
    train_draw_figure_gap_step_num=200,
    # to smooth the train loss
    train_loss_moving_average_step=3,
    # to show with multiple label, should a list[list]
    # value should be in metric_fields or `train_loss`
    train_loss_draw_combines=None,
    eval_loss_draw_combines=None,
    #### distribution params ####
    servers_file=None,
    # Default value 25Mb works fine.
    bucket_cap_mb=25,
    # "nccl" for GPU; "gloo" for GPU and CPU.
    backhand="gloo",
    # Usually you do NOT need to set it, as PAL Frame Would set for you in
    # the background.
    net_name=None,
    #### Eval params ####
    # Evaluation would be conducted every eval_gap_sample_num samples.
    # batch size during eval stage
    dev_file=None,
    test_files=None,
    eval_valid_file_extension=["pkl", "pydict", "json"],
    eval_batch_size=32,
    eval_num_workers_loading_data=2,
    eval_process_example_num_worker=1,
    eval_path_initial_model=None,
    # eval during train
    # whether eval during training
    # if this flag is true, then evaluator should be given as argument to trainer
    eval_during_training=False,
    eval_gap_step_num=None,
    eval_gap_sample_num=None,
    eval_gap_epoch_num=None,
    # main field in evaluation stage
    # this field must in return of evaluate.metric()
    metric_primary_field=None,
    metric_fields=[],
    # like F1 is large better, ppl is small better
    eval_value_is_large_better=None,
    #### Pred params ####
    # pred batch size
    pred_batch_size=32,
    pred_num_workers_loading_data=2,
    pred_path_initial_model=None,

    #### system params ####
    # Sets the number of threads used for intraop parallelism on CPU.
    num_threads_cpu=4,
    # None value denotes no limits on the maximum model size, or you should
    # set a value.
    automl_max_model_size=None,
    debug_level=1,  # debug=0, info=1, warning=2, error=3
    detect_anomaly=False,  # only for debugging.
    cudnn_deterministic=True,
    cudnn_benchmark=False,
    experiment_folder="work",
    # file name to save current best score
    best_eval_score_file_name = 'current_best_score.txt',
    run_tag=None,
    use_utc_time=True,
    folder_cache_meta_name=".meta.palframe.pkl")
