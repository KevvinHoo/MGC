# MGC
The Implementation About the MGC(`MGC:Maximum Gradient Compression in Distributed Deep Learning`)

# Setup
We used the GRACE, which is a gradient compression framework for PyTorch,TensorFlow and Keras, to implement our learning system. To start with our code, you should get the correct version of the software below:
```
Horovod == 0.18.2
PyTorch > 1.2
Torchvision
Tensorflow
tensorboardX
```

# To Start
We write two shellscipt to train the neural network such as ResNet,VGG,etc in a GPU cluster managered by SLURM. Run the follow command to start:
```
sbatch -N nodes_num -n tasks_num -p nodes_partition resnet.sh resnet_name ...
sbatch -N nodes_num -n tasks_num -p nodes_partition dist_train_cifar10.sh model_name ...
```


