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
        -v ${PWD}:/root/rggnet \
        -v /home/tienloc47/kitty_dataset:/root/kitti \
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

awesome-train-rggnet-3dstn:
	python commander.py train \
	--model_name "rggnet_3dstn"  \
	--gpu 1 \
	--clean_up False

eval-rggnet:
	export CUDA_VISIBLE_DEVICES=0 && python -m training.apps.evaluator process \
	--config_fp ../config/inference.yaml \
	--model_name rggnet \
	--res_fp report
