clean:
	docker rm -f $$(docker ps -qa)
build:
	docker build -t rggnet-docker .

run:
	docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="rggnet-experiment" \
	--user $(id -u):$(id -g) \
        -v ${PWD}:/root/rggnet \
	-v /mnt/disks/kitti/train_stereo:/root/kitti-train \
	-v /mnt/disks/kitti:/root/kitti-raw \
	rggnet-docker bash

test:
	echo "Running test ..."
	export CUDA_VISIBLE_DEVICES=0 && pytest training/tests/

test-tfrecords:
	export CUDA_VISIBLE_DEVICES=0 && python -m training.tests.test_tfrecords

# Below are new commands
awesome-train-vae:
	python commander.py train \
	--model_name "vae"  \
	--gpu 0 \
	--clean_up False

awesome-train-rggnet:
	python commander.py train \
	--model_name "rggnet"  \
	--gpu 0 \
	--clean_up False

awesome-train-rggnet-novae:
	python commander.py train \
	--model_name "rggnet_novae"  \
	--gpu 0 \
	--clean_up False

awesome-train-stereorggnet-novae:
	python commander.py train \
	--model_name "stereorggnet_novae"  \
	--gpu 0 \
	--clean_up False


awesome-train-rggnet-3dstn:
	python commander.py train \
	--model_name "rggnet_3dstn"  \
	--gpu 1 \
	--clean_up False

export-novae-model:
	export CUDA_VISIBLE_DEVICES= && python -m training.apps.production process \
	--config_fp ../config/rggnet_novae.yaml \
	--from_dir save/rggnet_novae_50/1679379005.712026 \
	--to_dir prod/rggnet_novae_50/ \
	--to_name best

export-stereo-novae-model:
	export CUDA_VISIBLE_DEVICES= && python -m training.apps.stereo_production process \
	--config_fp ../config/stereorggnet_novae.yaml \
	--from_dir save/stereorggnet_novae/1679291292.3442092 \
	--to_dir prod/stereo_rggnet_novae_50/ \
	--to_name best

eval-rggnet:
	export CUDA_VISIBLE_DEVICES=0 && python -m training.apps.evaluator process \
	--config_fp ../config/inference.yaml \
	--model_name rggnet_plus \
	--res_fp report

eval-rggnet-novae:
	export CUDA_VISIBLE_DEVICES=0 && python -m training.apps.evaluator process \
	--config_fp ../config/inference.yaml \
	--model_name rggnet_novae \
	--res_fp report_novae

eval-stereo-rggnet-novae:
	export CUDA_VISIBLE_DEVICES=0 && python -m training.apps.evaluator process \
	--config_fp ../config/stereo_inference.yaml \
	--model_name stereo_rggnet_novae \
	--res_fp report_stereo_novae