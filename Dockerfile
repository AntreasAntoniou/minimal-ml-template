FROM ghcr.io/bayeswatch/bwatch-tutorial:latest

RUN apt update
RUN apt install aptitude tree -y
RUN apt install fish -y
RUN echo y | pip install tabulate nvitop hydra_zen
RUN mamba install -c conda-forge starship jupyterlab black -y

RUN git lfs install
RUN git config --global credential.helper store
RUN git config --global --add safe.directory /workspaces/minimal-ml-template

RUN echo y | pip uninstall torch
RUN mamba uninstall pytorch -y

RUN mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN mamba install -c conda-forge timm accelerate datasets transformers -y

ENTRYPOINT ["/bin/bash"]
