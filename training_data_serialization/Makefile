# By default let's not use CUDA for data generation
create-data-64:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python data_generator.py run \
	--config_fp data_64.config.yaml

create-data-16:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python data_generator.py run \
	--config_fp data_16.config.yaml
