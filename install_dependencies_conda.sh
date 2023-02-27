conda install -y -c menpo opencv
pip install -r requirements.txt
echo "Downloading resnet_v2_50_2017_04_14"
mkdir resnet_v2_50_2017_04_14 & wget -qO- http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz |  tar -xvz -C resnet_v2_50_2017_04_14
