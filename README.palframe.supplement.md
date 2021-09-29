# Overview
PAL FRAME is a high-level frame built on Pytorch, specially customized for AutoML, Distributed Training, etc.
# Project Structure
A PAL FRAME project should be organized as follow:
```python
├── param.py                                                                                   
├── make_feature.py                                                                                   
├── make_feature_starter.py                                                                                   
├── train.py                                                                                   
├── train_starter.py                                                                                   
├── dataset.py                                                                                   
├── model.py                                                                                   
├── model_wrapper.py                                                                                   
├── model.py                                                                                   
├──[YOUR CUSTOMIZED FOLDERS]
│   ├── [YOUR CUSTOMIZED SUBFOLDERS]
│   │   └── ...
│   └── ...
```

### param.py
In this file you should define your own Param class by inheriting ParamBase class. Param class should include ALL configuration such as file path, training/validation/testing hyper parameter, etc.
### make_feature.py
In this file, you should write a script specifying how to preprocess the raw data and then store the processed data(as known as 'feature'). 
### model.py
In this file, you should define your model class, this model class can be any Pytorch-style model, you don't have to make any modification to an existing Pytorch model.
### train.py
In this file, you should write a script specifying how to train a model. To use PAL FRAME's support for Distributed Training, you are supposed to implement a Trainer class inheriting TrainerBase class. To implement Trainer class, you need to implement this abstract method: ```train_one_batch```.
### dataset.py
In this file, you should define your Pytorch-style dataset class, then wrap the dataset object with a iterable ojbect and then return this iterable object. This object should yield one data batch per iteration. 
For example, this iterable ojbect can be a Pytorch Dataloader instance, or you may define your own iterable objects. PAL FRAME will use the iterable object to loop through data.
### model_wrapper.py
In this file, you should write a ```ModelWrapper``` class by inheriting ```ModelWrapperBase``` class. Basically ```ModelWrapperBase``` has already done everything related to Distributed Training. In most cases, the only thing left is to implement this abstract method: ```evaluate_file``` in ```ModelWrapperBase```. 
### train_starter.py
This file should play a role of starting Distributed Training. You only need to call this function: ```starter.start_distributed_train``` in this script. Once you tells this function running options, it will do everything about Distributed Training for you. Note that ```starter.start_distributed_train``` will make the training process a background process automatically.
### make_feature_starter.py
This file should play a role of starting runing the data preprocessing program defined in make_feature.py in background mode distributedly. The only things to do in this script is to call ```starter.start_distributed_train```. The usage is the same as calling it in train_starter.py.


# PAL FRAME Architect
Generally, most backend work related to Distributed Training, Parameter Searching and other features has been already set, so the only remaining thing left is to implement some high-level interfaces. The most important component classes that need to be completed includes:
## 1. ```Param``` class
Param is a class that defines all configuration and hyper parameters for the model and training detail.
```Param``` class should inherit ```palframe.pytorch.estimator5.param.ParamBase```. 
To define your own Param class, you need to inherit ParamBase class first and may redefine some member variable in ParamBase class. You are also welcome to define new parameters when creating your ```Param```.
### 1.1 ```ParamBase``` class
```ParamBase``` is the Abstract Base Class(aka ABC) for all ```Param``` classes. 
In ```ParamBase``` class, many configuration parameters have been pre-defined, you can change these parameters by overriding these parameters.
Some usually used member variable in ParamBase:
```use_amp```(bool): Boolean varaible, represents whether Automatic Mixed Percision will be applied to the model.

```backhand```(str): String variable, represents which backend is used for distributed training, value of this variable should be either 'gloo'(for distributed CPU training) or 'nccl'(for distributed GPU training).

```servers_file```(str): String variable, should be a path of file which include a list of servers' ip. PAL Frame will use the ip to access the servers and use the servers' GPU and CPU for Distributed Training.

```net_name```(str): String variable, represents the name of network interface that is used to access internet/ethernet, you may find the name of network interfaces by using "ifconfig" command. Usually on MacOS, the net_name is "en0".

```train_files/vali_file/test_files```(list): List variable, represents a list of path that tells you where the data/features are. You should override these variables.

## 2. ```starter``` module
starter is a module that provides already-implemented interface for Distributed Running(including Distributed Data Preprocessing and Distributed Training). 
It is not necessary to override any code of this module. Just calling starter.start_distributed_train is enough in most cases.

## 3. ```ModelWrapper``` class
ModelWrapper is a wrapper for your model that enables you to train your model distributedly. 
You need to inherit ```palframe.pytorch.estimator5.model_wrapper.ModelWrapperBase``` class when you are creating your ModelWrapper.
### 3.1 ```ModelWrapperBase``` class
```ModelWrapperBase``` is a abstract class base that wrap your model and enable model to be trained on distributed mode( multi-GPU or multi-machine). 
This class provides a abstract method ```evaluate_file```. You should put the code about how to evaluate performance into this function when you are implementing your subclass of ModelWrapperBase.

## 4. ```Trainer``` class
Train class should define how the model is trained. 
```Trainer``` should inherit ```palframe.pytorch.estimator5.train.TrainerBase``` class when you are creating your own ```Trainer```.
### 4.1 ```TrainerBase``` class
TrainerBase is the abstract class base that defines how the train process happens. This class provides a abstract method ```train_one_batch```. You should put your code about how the train process happens per batch into this function. 