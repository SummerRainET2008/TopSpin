# How to run a toy example of intent detection
To make it easier for user to understand how PAL-frame works, we prepare a toy example in example/nlp/intent_detection. Below is the step.s (tested in environment ``python3.7``)
## 1.  GO TO {your-PAL-frame-folder}
    cd {your-PAL-frame-folder}

## 2. Data Preprocessing:
To prepare data, just run example/nlp/intent_detection/make_features.py as follow:

    python3.7 example/nlp/intent_detection/make_features.py

The processed features will be stored at ```{your-PAL-frame-folder}/feat/nlp/intent_detection```.

## 3. Train the Model in Debug Run Mode:
This mode is recommended when you want to debug. `In this mode, ONLY ONE GPU will be used`.
To train the model in debug run mode, just run

    nohup python3.7 example/nlp/intent_detection/train.py >intent_detection.log &
The trained model and traning information will be stored at ```{your-PAL-frame-folder}/work/run.nlp_example.intent_detection.{UTC time stamp}```. This folder contains all information related to training, such as:

   Training logs(in ```log``` folder) 

   Saved model checkpoints(in ```model``` folder)  

   Traning loss plot tensorboard(in ```tensorboard``` folder).


## 4. Train the Model in Auto Run Mode:
If you want to run this model with `MULTI-GPU(SINGLE MACHINE) mode or do parameter searching`, you can use auto run mode as follow: 

    nohup python3.7 example/nlp/intent_detection/train_auto_starter.py >intent_detection.1.log &

## 5. Train the Model in Distributed Run Mode:
If you want to run this model with `MULTI-MACHINE`, you can use distributed run mode as follow: 

    nohup python3.7 example/nlp/intent_detection/train_dist_starter.py >intent_detection.2.log &

Note that this example actually run a `SINGLE MACHINNE` training, if you want to use real `MULTI-MACHINE` training, you need to specify your own ```servers_file```.