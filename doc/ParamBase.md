# ParamBase

 1. Interface `__init__(self, run_tag: str, path_work_restored_training=None, experiment_folder="work", country_city="")` 
    * `run_tag`: user-defined name for his experiments.
    * `path_work_restored_training`: in the first running, this value is set as None; If the training is   stopped and user want to restore, this value is set as the path of the experiment.
    * `experiment_folder`: the default folder to store your experiment results.
    * `country_city`: set time zone for the Logger.info().

 1. Interface `generate_all_variants(self)`
    * Used in AutoML mode.

 1. Interface `clone(self, buff={})`
    * Return a clone.

 1. Attribute `self.seed = 0`

 1. Attribute `self.path_work`
    * `self.path_model`, `self.path_log`, `self.path_meta`, are set automatically

 1. Attribute `self.optimizer_name`

 1. Attribute `self.lr`

 1. Attribute `self.gpu_num`

 1. Attribute `self.use_gpu`

 1. Attribute `self.use_amp`

 1. Attribute `self.path_initial_model`

 1. Attribute `self.path_inference_model`

 1. Attribute `self.lr_decay_strategy`
    * Two learning rate decay strategies.
     * "linear": linear decay,
     * "stepwise_sgd": stepwise (traditional SGD style), lr_{epoch=n} = lr_{epoch=n-1} * decay_ratio, if decay_epochs == 1.

 1. Attribute `self.warmup_ratio = 0.`
    * For lr decay strategy 0: linear decay

 1. Attribute `self.ending_lr_ratio = 1e-5`
    * For lr decay strategy 0: linear decay
    * 1 means no lr decay.

 1. Attribute `self.stepwise_lr_decay_ratio = 0.1`
    * For lr decay strategy 1: step-wise_decay

 1. Attribute `self.stepwise_lr_decay_epochs = 30`

 1. Attribute `self.weight_decay = 0.01`
    * Such in RoBERTa, l2 weight decay is 0.01

 1. Attribute `self.param_norm = 1`
    * Default settings that work fine.

 1. Attribute `self.gpu_inference = 1`
    * Only support single-GPU inference. DataParallel is not applicable as
    it does not support module.parameters(), while in some important models,
    such as pytorch_transformers, they call module.parameters().

 1. Attribute `self.batch_dim = 0`

 1. Attribute `self.servers_file`

 1. Attribute `self.bucket_cap_mb = 25`
    * Default value 25Mb works fine.

 1. Attribute `self.backhand = "gloo"`
    * "nccl" for GPU; "gloo" for GPU and CPU.

 1. Attribute `self.net_name`
    * Usually you do NOT need to set it, as TopSpin Would set for you in the background.

 1. Attribute `self.variable_batch_size = {"<=30": 100, "<=128": 30}`
    * Example on one gpu

 1. Attribute `self.batch_size_one_gpu`

 1. Attribute `self.iter_num_update_optimizer = 1`
    * Gradient accumulation.

 1. Attribute `self.train_files`

 1. Attribute `self.vali_file`

 1. Attribute `self.test_files`

 1. Attribute `self.train_sample_num`

 1. Attribute `self.epoch_num`
    * A float is also allowed.

 1. Attribute `self.max_train_step`

 1. Attribute `self.eval_gap_sample_num`
    * Evaluation would be conducted every eval_gap_sample_num samples.

 1. Attribute `self.find_unused_parameters = True`

 1. Attribute `self.model_saved_num = 3`

 1. Attribute `self.num_threads_cpu = 4`
    * Sets the number of threads used for intraop parallelism on CPU.

 1. Attribute `self.num_workers_loading_data = 2`

 1. Attribute `self.automl_max_model_size = None`
    * None value denotes no limits on the maximum model size, or you should set a value.

 1. Attribute `self.true_gradient = False`
    * For the case that each GPU worker consumes different batch size.

 1. Attribute `self.debug_level = 1`
    * debug=0, info=1, warning=2, error=3

 1. Attribute `self.detect_anomaly = False`
    * Only for debugging.

 1. Attribute `self.cudnn_deterministic = True`

 1. Attribute `self.cudnn_benchmark = False`

 1. Attribute `self.draw_figure_frequency = 1000`

 1. Attribute `self.draw_figure_smooth_width = [1, 256]`

