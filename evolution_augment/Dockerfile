FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common curl git

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-distutils python3.10-distutils virtualenv

RUN virtualenv venv --python=$(which python3.10)

ENV VIRTUAL_ENV /workspace/venv

ENV PATH /workspace/venv/bin:$PATH

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

COPY ./requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

ARG USER
ARG UID
ARG GID
RUN groupadd -g $GID -o $USER
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $USER

COPY ./* /workspace/evolution_augment/

CMD [ "/workspace/venv/bin/python",  "-m",  "evolution_augment" ]