# Introduction

This is a repository containing generic modules for neural networks and examples of use of PyTorch Lightning and Hydra libraries/frameworks, as well as dummy files for implementing your own neural networks.
As a reminder, PyTorch Lightning and Hydra are in general independent, and one can use only PyTorch Lightning on its own.

A brief summary of Pytorch Lightning and Hydra libraries:

## PyTorch Lightning 

It's a library built on top of PyTorch which allows writing models and training them with less boilerplate code, because it take care of training the neural network. The resulting code is also more modular. The neural network models must contain the functions:

`forward(), configure_optimizers(), training_step(), validation_step()`. The `training_step()` function must return the "loss".

A minimal example is shown on the website:

https://www.pytorchlightning.ai/

After implementing these function inside a class inheriting from `pl.LightningModule` (see example_classifier/example_classifier.py), the training code is simplified and looks like this:

```
trainer = pl.Trainer(gpus=4, max_epochs=1500, max_time={"days": 0, "hours": 0,"minutes": 5})
```

```
trainer.fit(model, train_dataloader, val_dataloader)
```

Installation instructions:

```
pip install pytorch-lightning
```
  

## Hydra

Hydra is a library that allows dynamic composition of configurations. The configurations are written in config files and can be overwritten in the command line.

> https://hydra.cc/

Examples are also shown here
> https://github.com/facebookresearch/hydra/tree/master/examples

Installation instructions:
```
pip install hydra-core --upgrade
```

# Example for training a MNIST classifier
To start training an example classifier, run
```
python train_hydra.py 
```
The `conf` directory includes directories `db`, `network`, `opt` and `pl_trainer`. These are used to define the network, database and several training hyperparameters. You can also define or override these parameters using 

Inside the file `conf/pl_trainer/default.yaml`, for example, is the default training length of `max_epochs: 15`. If you want to change this parameter, or any other, you can do it like so:

```
python train_hydra.py +network=classifier ++pl_trainer.max_epochs=2
```

The configuration files are usually stored in the `conf` folder. The defaults are stored in `conf/config.yaml`.

You can check out some example conf files in the repositories `imednet/branch rimednet_lightning` or `intentnet/` branch `alt`.

# Steps to make your own project

1. Add a new directory `my_project` into the `projects` directory.
2. Inside this folder, create files `my_network.py`, which takes care of the network training and `my_datamodule.py`, which defines how PyTorch Lightning module `LightningDataModule` loads data. You should also write a `my_dataset.py` file that defines how you read your dataset files and use it inside the data module. There are some examples in the `dataset.py` file.
3. Add YAML configuration files `conf/network/my_network.yaml` and `conf/db/my_dataset.yaml`. Here you can define the target network: 
    ```
    _target_: .projects.my_project.my_network.MyNetwork
    ```
    and the target database:
    ```
    _target_: .projects.my_project.my_datamodule.MyDataset
    ```
    as well as any desired arguments (an example of arguments passing is shown in the next section).
4. Then you can run `train_hydra.py` or `test_hydra.py` (they are separated for convenience). You can also specify parameters (such as the desired network) and also override parameters when running (using the ++ sign):
    ```
    python train_hydra.py +network=my_network ++network.network.args.dev=cpu ++db.db.args.batch_size=30
    ```

5. (Recommended) It is also highly useful to define `training_epoch_end`, `validation_epoch_end` and `test_epoch_end` functions in `my_network.py` file, since you can then work with the outputs from all previous training, validation or test steps (for example if you want to calculate accuracy across all batches). See `projects/intentnet/intentnet.py` for an example use of these functions.

See the files in the `dummy_example` directory, as well as dummy YAML files `conf/db/dummy_dataset.yaml` and `conf/network/dummy_network.yaml` for more detailed instructions.

Two working examples are also available in `projects/intentnet` and `projects/talosnet` directories. These utilize early stopping using `self.log()` function.

The `intentnet` project trains a network that classifies input RGBD videos into a limited set of labels, while the `talosnet` projects predicts the goal location of the human hand based on an input sequence of past hand position measurements.

## About the `conf` folder

Inside this folder is the file `config.yaml` which contains default configuration for the database loader and model, in this repository for example it is:
```
defaults:
  - db/mnist
  - network/classifier
```
This means that Hydra will try to find files `conf/db/mnist.yaml`. Inside this file it must be written for example
```
_target_: .example_classifier.example_classifier.LitClassifier
```

This means that hydra will try to instantiate the module `LitClassifier`, and pass any arguments that are written below `_target_`, for example
```
_target_: .example_classifier.example_classifier.LitClassifier
args:
    arg_1: 10
    arg_2: 20
```
Should you wish to use a different neural network, make a new file `conf/network/my_network.yaml` and in it specify a different `_target_` as well as any arguments.

If you want to train example classifier on MNIST data, you would run :

```
python train_hydra.py 
```

If you wanted to replace the `LitClassifier` example module with your own, you would run:

```
python train_hydra.py +network=my_network
```

Then hydra would check the `conf/network/my_network.yaml` file and instantiate the \_target\_ module.

If training, i.e., RIMEDNet network on video-trajectory database and saving outputs to `outputs/tmp` directory, you can use:

```
python train_hydra.py +db=video_traj_dataset +network=intentnet ++hydra.run.dir=outputs/my_output_dir
```
