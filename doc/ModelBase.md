# ModelBase

 1. Interface `__init__(self, param: ParamBase)`

 1. Interface `forward(self, *args, **kwargs)`:   
    * Users need to define to return a loss given a list of input tensors.

 1. Interface `set_device(self, device: torch.device)`
    * User-defined GPU or CPU. Usually you don't need to set it.

 1. Interface `load_model(self, model_file)`
    * Built-in implementation.
    
 1. Interface `load_model_from_folder(self)`
    * Built-in implementation.

```