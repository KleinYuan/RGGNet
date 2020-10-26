run-fs-mode-cpu-rggnet:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python main.py run \
	--config_fp config.yaml \
	--model_name "rggnet" \
	--profile_export "profile"

run-fs-mode-cpu-rggnet-novae:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.'&& python main.py run \
	--config_fp config.yaml \
	--model_name "rggnet_novae" \
	--profile_export "profile"

run-fs-mode-gpu-rggnet:
	export PYTHONPATH='.' &&  python main.py run \
	--config_fp config.yaml \
	--model_name "rggnet" \
	--profile_export "profile"

run-fs-mode-gpu-rggnet-novae:
	export PYTHONPATH='.' && python main.py run \
	--config_fp config.yaml \
	--model_name "rggnet_novae" \
	--profile_export "profile"

run-fs-mode-gpu-rggnet-plus:
	export PYTHONPATH='.' &&  python main.py run \
	--config_fp config.yaml \
	--model_name "rggnet_plus" \
	--profile_export "profile"

run-fs-mode-gpu-all:
	export PYTHONPATH='.' && python main.py run_all \
	--config_fp config.yaml
