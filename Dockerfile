FROM ghcr.io/bayeswatch/bwatch-tutorial:latest

RUN apt update
RUN apt install aptitude tree -y
RUN echo yes | pip install tabulate nvitop hydra_zen

RUN git lfs install
RUN git config --global credential.helper store
RUN git config --global --add safe.directory /workspaces/minimal-ml-template

ENTRYPOINT ["/bin/bash"]
