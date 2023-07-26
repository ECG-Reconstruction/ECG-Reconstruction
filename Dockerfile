# syntax=docker/dockerfile:1

# To build the docker image:
# docker build --tag IMAGE-NAME DIR-WITH-DOCKERFILE

# To run the docker image:
# docker run \
#     --runtime nvidia --gpus all \
#     --shm-size 4G \
#     [--interactive --tty] \
#     [--rm] \
#     [--publish 8888:8888] \
#     IMAGE-NAME \
#     COMMAND ...
#
# Explanation:
# - `--runtime nvidia --gpus all` enables GPU support and allows the container to use all GPUs.
# - `--shm-size 4G` increases the shared memory size. This is required for PyTorch dataloaders, which use shared memory for inter-process data transfer.
# - If the command is interactive, then use the `--interactive` and `--tty` flags.
# - If `--rm` is used, then the container is removed after the command finishes.
# - If `--publish 8888:8888` is used, then port 8888 inside the container is forwarded to port 8888 on the host.

# Example: To start a JupyterLab server:
# docker run \
#     --runtime nvidia --gpus all \
#     --shm-size 4G
#     --rm \
#     --publish 8888:8888 \
#     yunhao-qian/ecg-reconstruction \
#     jupyter lab --allow-root --ip 0.0.0.0
#
# Explanation:
# - By default, the server refuses to run as root. Use `--allow-root` to override this behavior.
# - By default, the server listens on localhost, which is not accessible outside the container. Use `--ip 0.0.0.0` to listen on all IPs.

FROM ubuntu:22.04

# Upgrade system packages, install curl, and clean caches.
RUN apt-get update && \
    apt-get --yes upgrade && \
    apt-get --yes install --no-install-recommends ca-certificates curl git && \
    apt-get --yes clean

# Install Mambaforge.
RUN curl --location --output ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Upgrade and install Conda packages.
COPY docs/linux_installation/cuda_environment.yaml environment.yml
RUN /opt/conda/bin/mamba update --yes conda mamba && \
    /opt/conda/bin/mamba env update --file environment.yml --name base && \
    rm environment.yml

# Clean caches.
RUN /opt/conda/bin/python -m pip cache purge && \
    /opt/conda/bin/mamba clean --yes --all

# Add Conda to `PATH`, which also changes the default `python` to the conda version.
ENV PATH /opt/conda/bin:$PATH
