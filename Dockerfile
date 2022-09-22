FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git python3 python3-pip vim swig
RUN python3 -m pip install "ray[rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl" 
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa"
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN pip3 install box2d wandb dnc opt_einsum einops
RUN git clone git@github.com:smorad/popgym.git
RUN pip3 install -e popgym
