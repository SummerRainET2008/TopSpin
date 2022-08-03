# -*- coding: utf-8 -*-
#author: Xuan Zhou
"""
palframe command tools, including start_dist_train ...
"""

import sys, os
from palframe import nlp
from palframe.nlp import Logger
from palframe.nlp import load_module_from_full_path
from functools import partial
from argparse import ArgumentParser


def list_parse(list_str):
  l = eval(list_str)
  assert isinstance(l, list), f'{list_str} cannot parse as list'
  return l


# command params for starting distributed train
START_DIST_TRAIN_PARAMS = [
    ('servers_file', str, 'servers file,e.g. ip1,ip2,ip3'),
    ('train_files', str, 'train files location'),
    ('vali_files', str, 'validation files location'),
    ('test_files', str, 'test files location'),
    ('path_initial_model', str, 'inital checkpoint'),
    ('gpus', list_parse, 'gpus, such as `[0,1,2]`'),
    ('epoch_num', int, 'train stop condition: total epoch'),
    ('max_train_step', int, 'train stop condition:: max train step'),
    ('model_saved_num', int, 'max checkpoint num to save'),
    ('iter_num_update_optimizer', int, 'gradient accumulation num'),
    ('batch_size', int, 'batch size')
]


def parser_args():
  parser = ArgumentParser(description='palframe command tools')
  subparser = parser.add_subparsers(help='sub-command')

  # global param
  global_parser = ArgumentParser(add_help=False)
  global_parser.add_argument('--debug_level',
                             type=int,
                             default=1,
                             help='logger level')

  subparser.add_parser = partial(subparser.add_parser, parents=[global_parser])

  # start distributed train
  start_dist_train_parser = subparser.add_parser(
      'start_dist_train', help='start a dsitributed trainning')
  start_dist_train_parser.add_argument('--train_script_name',
                                       default='train.py',
                                       help='name of train.py')
  start_dist_train_parser.add_argument('--trainer_cls_name',
                                       default='Trainer',
                                       help='class name of trainer')
  start_dist_train_parser.add_argument('--param_script_name',
                                       default='param.py',
                                       help='name of param.py')
  start_dist_train_parser.add_argument('--param_cls_name',
                                       default='Param',
                                       help='class name of param')
  start_dist_train_parser.add_argument('--project_dir',
                                       required=True,
                                       help='location of project path')
  start_dist_train_parser.add_argument('--extra_run_tag',
                                       default=None,
                                       help='extra run tag')
  # other params to add
  for p_name, p_type, p_help in START_DIST_TRAIN_PARAMS:
    start_dist_train_parser.add_argument(f"--{p_name}",
                                         type=p_type,
                                         default=None,
                                         help=p_help)

  start_dist_train_parser.set_defaults(func=start_dist_train)

  # stop dist train
  stop_dist_train_parser = subparser.add_parser(
      'stop_dist_train', help='stop a dsitributed trainning')
  stop_dist_train_parser.add_argument('--path_work', default=None)
  stop_dist_train_parser.add_argument('--servers',
                                      default=None,
                                      help="ip1,ip2,ip3")

  stop_dist_train_parser.add_argument('--servers_file', default=None)

  stop_dist_train_parser.set_defaults(func=stop_dist_train)

  return parser.parse_args()


def param_init_fn_decorator(run_tag):
  """
    modify init funciton's run_tag keyword
    """
  def wrapper1(fn):
    def wrapper2(*args, **kwargs):
      kwargs['run_tag'] = run_tag
      fn(*args, **kwargs)

    return wrapper2

  return wrapper1


def start_dist_train(args):
  """
    launch a distributed training
    """
  train_script_name = args.train_script_name
  param_script_name = args.param_script_name
  # trainer_cls_name = args.trainer_cls_name
  extra_run_tag = args.extra_run_tag
  param_cls_name = args.param_cls_name

  project_dir = args.project_dir

  train_script_path = os.path.join(project_dir, train_script_name)
  param_script_path = os.path.join(project_dir, param_script_name)

  assert os.path.exists(train_script_path), train_script_path
  assert os.path.exists(param_script_path), param_script_path

  # get param cls
  param_module = load_module_from_full_path(param_script_path)
  Param_cls = getattr(param_module, param_cls_name)

  # modify run_tag in param class's __init__ signature
  # by using a decorator
  import inspect
  sig = inspect.signature(Param_cls.__init__)
  run_tag = sig.parameters['run_tag'].default
  if extra_run_tag is not None:
    run_tag = run_tag + '.' + extra_run_tag
  Param_cls.__init__ = param_init_fn_decorator(run_tag)(Param_cls.__init__)

  # get param object and modify some param from command
  param_obj = Param_cls.get_instance()

  start_dist_train_param_names = [p[0] for p in START_DIST_TRAIN_PARAMS]
  for name in start_dist_train_param_names:
    value = getattr(args, name)
    if value is not None:
      setattr(param_obj, name, value)
  # print(param_obj)
  # print(train_script_path)
  # print(fg)
  # launch train
  from palframe.pytorch.estimator6 import starter
  starter.start_distributed_train(param_obj, train_script_path)


def stop_dist_train(args):
  """
    stop dist train, code copy from stopper.py
    """
  from palframe.pytorch.estimator6 import starter
  import re
  if not nlp.is_none_or_empty(args.path_work):
    path_work = args.path_work
    if "run_id_" in path_work:
      run_id = re.compile(r"run_id_(.*\d+)").findall(path_work)[0]
      starter.stop_train(run_id)
    else:
      starter.stop_distributed_train(path_work)

  elif not nlp.is_none_or_empty(args.servers):
    for ip in args.servers.split(","):
      starter.clear_server(ip)

  elif not nlp.is_none_or_empty(args.servers_file):
    for ip in open(args.servers_file).read().replace(",", " ").split():
      starter.clear_server(ip)


def main():
  args = parser_args()
  debug_level = args.debug_level
  Logger.set_level(debug_level)
  cmd = " ".join(sys.argv)
  args.cmd = f"palframe {cmd}"
  Logger.info(f"launch command:\n {args.cmd},\n params: {vars(args)}")
  args.func(args)


if __name__ == "__main__":
  main()
