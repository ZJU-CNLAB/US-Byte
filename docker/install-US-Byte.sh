#!/bin/bash
# Date：2023-03-07
# Author: Create by Yunqi Gao
# Description: This script function installs ByteScheduler with US-Byte.
# Version： 1.0

cd && cd byteps/bytescheduler/bytescheduler/common && mv /root/code/US-Byte/docker/file/bytecore.py bytecore.py &&
mv /root/code/US-Byte/docker/file/bytetask.py bytetask.py && cd &&

cd byteps/bytescheduler/bytescheduler/pytorch && mv /root/code/US-Byte/docker/file/horovod.py horovod.py &&
mv /root/code/US-Byte/docker/file/horovod_task.py horovod_task.py && cd &&

cd /root/code/US-Byte/byteps/bytescheduler && python setup.py install &&

pip install mpi4py
