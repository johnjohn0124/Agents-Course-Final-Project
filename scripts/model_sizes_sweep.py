import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/seobrien/.cache/'

from omegaconf import OmegaConf
from ilql_utils import train_ilql
import json
from collections import Counter

import yaml

if __name__=="__main__":


    SEEDS = [
        # 0,
        # 1,
        2,
        # 3,
        # 4
    ]
    MODEL_NAMES = [
        # 'openai-community/gpt2-xl',
        'openai-community/gpt2-large',
        'openai-community/gpt2-medium',
        'openai-community/gpt2',
    ]

    config_path = './configs/ilql/default.yaml'
    config = OmegaConf.load(config_path)

    config.saving.save_basedir = config.saving.save_basedir.format(task=config.task)
    config.data_path = config.data_path.format(task=config.task)

    config.run_group_name = "model-size-sweep"

    config.training.train_data_diet = 10000


    for model_name in MODEL_NAMES:
        for seed in SEEDS:

            config.run_name = f"model-{model_name.replace('/', '__')}/seed-{seed}"
            config.saving.save_dir = os.path.join(config.saving.save_basedir,
                                        config.run_group_name,
                                        config.run_name)

            config.training.seed = seed
            config.model_name = model_name

            config.training.n_epochs = int(20000 // config.training.train_data_diet)

            os.makedirs(config.saving.save_dir, exist_ok=True)

            with open(os.path.join(config.saving.save_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(config=config, f=f)

            train_ilql(config)

