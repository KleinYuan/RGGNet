# Summary

This is a tutorial to guide you to setup an ubuntu machine. You are welcome!
 
- [X] Refer: https://gist.github.com/matheustguimaraes/43e0b65aa534db4df2918f835b9b361d

# 11 Steps

#### Step1 Install NVIDIA DRIVER
sudo apt-get upgrade
sudo apt-get update
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-410

#### Step2 Reboot
reboot

##### Step3 Check NVIDIA DRIVER
nvidia-smi

#### Step4 Install CUDA 10.0
- [X] Download Linux/x86_64/Ubuntu/16.04/runfile(local): https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

`
sudo sh cuda_10.0.130_410.48_linux.run
`
```
    Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81? n
    Install the CUDA 10.0 Toolkit? y
    Do you want to install a symbolic link at /usr/local/cuda? y
    Install the CUDA 10.0 Samples? y
```

#### Step5 Test CUDA 

```
cd /usr/local/cuda/samples
sudo make -k
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
./deviceQuery
```

#### Step6 Install cuDNN v7.4.2 (Dec 14, 2018), for CUDA 10.0

Download the following:

- [X] cuDNN Runtime Library for Ubuntu16.04 (Deb)
- [X] cuDNN Developer Library for Ubuntu16.04 (Deb)
- [X] cuDNN Code Samples and User Guide for Ubuntu16.04 (Deb)
- [X] sudo dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb
- [X] sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb 
- [X] sudo dpkg -i libcudnn7-doc_7.4.2.24-1+cuda10.0_amd64.deb

The add those to ~/.bashrc

```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
source ~/.bashrc
```

#### Step7 Test cuDNN

```
cd Desktop/cudnn_samples_v7/mnistCUDNN/
make clean && make
./mnistCUDNN
```

#### Step8 Install Anaconda Anaconda3-2020.02-Linux-x86_64.sh

- [X] Follow https://docs.anaconda.com/anaconda/install/linux/

```
source ~/.bashrc
conda config --set auto_activate_base False
```

#### Step9 Create an python 3.7 environment

```
conda create -n py3 python==3.7
```

#### Step10 Install tensorflow-gpu 1.14

```
conda activate py3
pip install tensorflow-gpu==1.14
```

#### Step11 Test GPU availability

```
python
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
```
~> You shall see True


