# Primer MNIST klasifikatorja

# Prvi korak je, da dobis podatke v obliki data-loaderja. 

# G
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
mnist_module = MNISTDataModule()
mnist_module.setup()
train_d, val_d = mnist_module.dataset_train,mnist_module.dataset_val

train_dl = torch.utils.data.DataLoader(train_d)
val_dl = torch.utils.data.DataLoader(val_d)


# Drugi korak je definiranje 

from generic_nn_modules.example_classifier import LitClassifier
model = LitClassifier()
