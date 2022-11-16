FROM ghcr.io/antreasantoniou/tali-collector-dev:0.2.0

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN apt update
RUN apt install aptitude tree -y
RUN echo yes | pip install tabulate nvitop pytorchvideo hydra_zen python-dotenv
RUN mamba update -c conda-forge ffmpeg starship itables -y
RUN mamba install -c conda-forge pytorch-lightning transformers timm -y
RUN mamba install -c conda-forge git-lfs wandb -y
RUN git lfs install
RUN git config --global credential.helper store
RUN git config --global --add safe.directory /workspaces/minimal-ml-template

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]