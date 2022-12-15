FROM ghcr.io/bayeswatch/bwatch-tutorial:latest

RUN apt update
RUN apt upgrade -y

RUN apt install aptitude tree -y
RUN apt install fish -y
RUN echo y | pip install tabulate nvitop hydra_zen
RUN mamba install -c conda-forge starship jupyterlab black -y

RUN git lfs install
RUN git config --global credential.helper store

RUN echo y | pip uninstall torch
RUN mamba uninstall pytorch -y

RUN mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN mamba install -c conda-forge timm accelerate datasets transformers -y

RUN apt install kubectl -y
RUN apt install google-cloud-sdk-gke-gcloud-auth-plugin -y
ENV USE_GKE_GCLOUD_AUTH_PLUGIN True

RUN echo y | pip install git+https://github.com/BayesWatch/bwatchcompute@main

RUN rm -rf /app/
ADD . /app/

RUN git config --global --add safe.directory /app/

RUN echo y | pip install /app/

ENTRYPOINT ["/bin/bash"]
