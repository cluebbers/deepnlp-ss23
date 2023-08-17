#!/usr/bin/env bash

if ! [ -x "$(command -v conda)" ]
then
    echo "Please install conda before running setup!"
    exit
fi

conda create -n dnlp python=3.8
conda activate dnlp
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7
pip install tensorboard
pip install torch-tb-profiler
pip install optuna