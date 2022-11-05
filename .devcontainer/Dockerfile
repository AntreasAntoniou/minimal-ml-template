FROM ghcr.io/antreasantoniou/tali-collector-dev:0.2.0

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN apt update
RUN apt install aptitude tree -y
RUN echo yes | pip install tabulate nvitop pytorchvideo
RUN mamba update -c conda-forge ffmpeg starship itables -y

ENTRYPOINT ["/bin/bash"]