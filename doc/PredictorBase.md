# PredictorBase

 1. Interface `__init__(self, model: ModelBase)`
    * Built-in implementation.

 1. Interface `load_model(self)`
    * Built-in implementation.

 1. Interface `predict(self, *batch)`
    * Built-in implementation.

 1. Interface `evaluate_file(self, data_file) -> float`
    * Users need to define task-specific measures. 
    * Note that, the result is `the smaller, the better`. 
    * Generally you need to return a negated measure value. 
