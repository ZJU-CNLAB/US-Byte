# US-Byte: An Efficient Communication Mechanism for Scheduling Unequal-sized Tensor Blocks in Distributed Deep Learning #  
## Introduction ##
This repository contains the codes of the US-Byte paper accepted by *IEEE TPDS*. US-Byte is a communication mechanism for scheduling unequal-sized tensor blocks implemented on the PyTorch framework. US-Byte outperforms the state-of-the-art communication scheduling mechanisms ByteScheduler and WFBP.  
<div align=center><img src="system%20architecture.png" width="500"/></div> 

## Installation ##
### Prerequisites ###
We highly recommend using Docker images for experimenting. The following prerequisites shoud be installed in order to use the Docker images for this repository:  
* Ubuntu 18.04  
* CUDA >= 9.0  
* Docker 20.10.18  
* NVIDIA docker
### Data Processing ###
You can unzip the Cifar100 and ImageNet2012 in /data folder and run the following scripts in Python3 to prepare the dataset:  
```
python ./scripts/process_cifar100.py  
python ./scripts/process_imagenet2012.py  
```
### Quick Start ###
You can download this code to /root/code folder and run the following scripts:  
```
cd /root/code/US-Byte/docker  
docker build -t us-byte-pytorch:v1 --no-cache -f US_Byte_pytorch .  
nvidia-docker run -it --net=host --shm-size=32768m -v /data:/data -v /root/code:/root/code us-byte-pytorch:v1 bash  
cd /root/code/US-Byte/docker  
./install-US-Byte.sh  
cd /root/code/US-Byte  
chmod 777 ./dist.sh  
dnn=vgg16 nworkers=4 ./dist.sh
```  
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the VGG16 model with the ImageNet2012 dataset using the US-Byte mechanism. The partition size is obtained by Bayesian optimization, and you can tune it manually according to [ByteScheduler's communication scheduling](https://github.com/bytedance/byteps/blob/bytescheduler/bytescheduler/docs/scheduling.md).
