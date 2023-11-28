# -*- coding: utf-8 -*-
#author: Xuan Zhou
"""
palframe command tools, including start_dist_train ...
"""

import importlib
import sys, os, time, traceback
from src.topspin import \
  nlp
from src.topspin.nlp import Logger
from functools import partial
from argparse import ArgumentParser

abs_path_dir = os.path.dirname(__file__)
template_dir = os.path.join(os.path.dirname(abs_path_dir), 'pytorch',
                            'estimator7', 'templates')


def list_parse(list_str):
  l = eval(list_str)
  assert isinstance(l, list), f'{list_str} cannot parse as list'
  return l


# command params for starting distributed train
START_DIST_TRAIN_PARAMS = [
    ('run_tag', str, 'work_path_name'),
    ('path_work_restored_training', str, 'path_work of restored training'),
    ('servers_file', str, 'servers file,e.g. ip1,ip2,ip3'),
    ('train_files', str, 'train files location'),
    ('dev_files', str, 'validation files location'),
    ('test_files', str, 'test files location'),
    ('train_path_initial_model', str, 'train initial checkpoint'),
    ('gpus', list_parse, 'gpus, such as `[0,1,2]`'),
    ('gpu_num', int, 'gpu num'),
    ('epoch_num', int, 'train stop condition: total epoch'),
    ('max_train_step', int, 'train stop condition:: max train step'),
    ('model_saved_num', int, 'max checkpoint num to save'),
    ('iter_num_update_optimizer', int, 'gradient accumulation num'),
    ('train_batch_size', int, 'train_batch size'),
    ('eval_batch_size', int, 'eval batch size'),
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

  # start new project
  create_project_parser = subparser.add_parser('init',
                                               help='init a new project')
  create_project_parser.set_defaults(func=create_new_project)

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
  start_dist_train_parser.add_argument(
      '--project_dir',
      required=True,
      help='location of project path, default is same as train_script_name')
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

  # create folder meta
  create_folder_meta_parser = subparser.add_parser('create_folder_meta',
                                                   help='create folder meta')

  create_folder_meta_parser.add_argument('feat_path', help='folder path')
  create_folder_meta_parser.add_argument('--valid_file_extension',
                                         default=["pkl", "pydict", 'json'],
                                         type=list_parse,
                                         help='valid file extensions')
  create_folder_meta_parser.add_argument('--meta_file_name',
                                         default=".meta.palframe.pkl",
                                         help='meta file name')

  create_folder_meta_parser.set_defaults(func=create_folder_meta)

  start_server_parser = subparser.add_parser(
      'start_server', help='start a flask server to predict')
  start_server_parser.add_argument(
      '--worker_dist',
      type=str,
      default='{-1:1}',
      help='worker distribution dict, a json file path or json str,'
      'for example: `{0:1,1:1}` mean cuda:0 has one worker, cuda:1 has one worker'
  )
  start_server_parser.add_argument(
      '--lazy_start',
      action='store_true',
      default=False,
      help=
      'decide if use lazy start, if so, model will not start when service is started'
  )

  start_server_parser.add_argument('--param_script_name',
                                   default='param.py',
                                   help='name of param.py')
  start_server_parser.add_argument('--param_cls_name',
                                   default='Param',
                                   help='class name of Param class')

  start_server_parser.add_argument('--model_script_name',
                                   default='model.py',
                                   help='name of model.py')
  start_server_parser.add_argument('--model_cls_name',
                                   default='Model',
                                   help='class name of Model class')

  start_server_parser.add_argument('--predictor_script_name',
                                   default='predict.py',
                                   help='name of model.py')
  start_server_parser.add_argument('--predictor_cls_name',
                                   default='Predictor',
                                   help='class name of Prodictor class')

  start_server_parser.add_argument(
      '--model_checkpoint',
      default=None,
      help='model checkpoint path sent to predictor')

  start_server_parser.add_argument(
      '--restart_with_cpu',
      action='store_true',
      default=False,
      help='by default, model will restart refer to worker_dist argument,'
      ' this argument forces restart with cpu')
  start_server_parser.add_argument('--debug',
                                   default=False,
                                   action='store_true',
                                   help='debug mode')
  start_server_parser.add_argument('--port',
                                   type=int,
                                   default=5018,
                                   help='specify port')
  start_server_parser.add_argument('--auto_exit_time',
                                   type=int,
                                   default=1800,
                                   help='auto exit time')
  start_server_parser.add_argument(
      '--submit_desc',
      type=str,
      default=None,
      help='a json file path, which describe the data example format to submit')

  start_server_parser.add_argument('--save_request_data',
                                   default=False,
                                   action='store_true',
                                   help='whether save requests data')
  start_server_parser.add_argument(
      '--max_keep',
      default=5,
      type=int,
      help='max data to keep with error or normal data')

  start_server_parser.set_defaults(func=start_server)

  return parser.parse_args()


def create_folder_meta(args):
  feat_path = args.feat_path
  valid_file_extension = args.valid_file_extension
  meta_file_name = args.meta_file_name
  from src.topspin.pytorch import FolderMetaCache
  FolderMetaCache.create_meta_file(feat_path, valid_file_extension,
                                   meta_file_name)


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
  # extra_run_tag = args.extra_run_tag
  param_cls_name = args.param_cls_name

  project_dir = args.project_dir

  train_script_path = os.path.relpath(
      os.path.join(project_dir, train_script_name))
  param_script_path = os.path.relpath(
      os.path.join(project_dir, param_script_name))
  assert os.path.exists(train_script_path), train_script_path
  assert os.path.exists(param_script_path), param_script_path

  # get param cls
  param_module = importlib.import_module(
      param_script_path.replace(".py", "").replace('/', '.'))
  Param_cls = getattr(param_module, param_cls_name)

  # modify run_tag in param class's __init__ signature
  # by using a decorator
  # import inspect
  # sig = inspect.signature(Param_cls.__init__)
  # run_tag = sig.parameters['run_tag'].default
  # if extra_run_tag is not None:
  # run_tag = run_tag + '.' + extra_run_tag
  # Param_cls.__init__ = param_init_fn_decorator(run_tag)(Param_cls.__init__)

  # get param object and modify some param from command
  param_obj = Param_cls.get_instance()

  start_dist_train_param_names = [p[0] for p in START_DIST_TRAIN_PARAMS]
  for name in start_dist_train_param_names:
    value = getattr(args, name)
    if value is not None:
      setattr(param_obj, name, value)

  # if param_obj.
  #   param_obj.create_restart_work_path_name()
  from src.topspin.pytorch import starter
  starter.start_distributed_train(param_obj, train_script_path)


def stop_dist_train(args):
  """
    stop dist train, code copy from stopper.py
    """
  from src.topspin.pytorch.estimator6 import \
    starter
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


def create_new_project(args):
  """
    create new project 
    :param args:
    :return:
    """
  args = vars(args)
  # mutual cmd
  while True:
    project_name = input('project name: ')
    if project_name:
      args['project_name'] = project_name
      break
    else:
      print('Project name cannot be empty, please re-enter ')
  author_name = input('author name: ')
  args['author'] = author_name

  author_email = input('author email: ')
  args['email'] = author_email

  args['time'] = time.strftime("%Y/%m/%d %H:%M", time.localtime())

  distribute_from_templates(args)


def distribute_from_templates(args):
  """
    :param args: 
    :return:
    """

  def _render_one_file(path_sorce, path_target, args):
    try:
      with open(path_sorce, 'r', encoding='utf-8') as f:
        r = f.read()
      r_render = r.replace(r'{{time}}',args['time']).\
          replace(r'{{author}}',args['author']).\
          replace(r'{{email}}',args['email']).\
          replace(r'{{project}}', args['project_name'])
      with open(path_target, 'w', encoding='utf-8') as f:
        f.write(r_render)
    except:
      print(traceback.print_exc())
      print('当前渲染出错的路径', path_sorce, path_target)

  path_project = os.path.join(os.getcwd(), args['project_name'])
  if os.path.exists(path_project):
    raise ValueError('the project path  already exists')
  os.mkdir(path_project)

  paths_total = [
      os.path.join(template_dir, 'param.py'),
      os.path.join(template_dir, 'model.py'),
      os.path.join(template_dir, 'train.py'),
      os.path.join(template_dir, 'evaluate.py'),
      os.path.join(template_dir, 'make_feature.py'),
  ]

  for path in paths_total:
    _render_one_file(path, os.path.join(path_project, os.path.basename(path)),
                     args)


def start_server(args):
  # 启动预测服务
  from src.topspin.pytorch import start_server
  start_server(args)


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
