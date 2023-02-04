FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0

RUN conda init bash
RUN conda create -n main python=3.10 -y
RUN echo "conda activate main" >> ~/.bashrc

SHELL ["conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN apt update
RUN apt upgrade -y

RUN apt install aptitude tree -y
RUN apt install fish -y
RUN echo y | pip install tabulate nvitop hydra_zen wandb --upgrade
RUN conda install mamba -y 
RUN mamba install -c conda-forge starship jupyterlab black git-lfs -y

RUN git lfs install
RUN git config --global credential.helper store

RUN mamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN mamba install -c conda-forge timm accelerate datasets transformers -y
RUN mamba install -c conda-forge orjson -y

RUN apt install kubectl -y
RUN apt install google-cloud-sdk-gke-gcloud-auth-plugin -y
ENV USE_GKE_GCLOUD_AUTH_PLUGIN True

RUN echo y | pip install git+https://github.com/BayesWatch/bwatchcompute@main

RUN rm -rf /app/
ADD . /app/

RUN git config --global --add safe.directory /app/

RUN echo y | pip install /app/

ENTRYPOINT ["/bin/bash"]
