FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

WORKDIR /interactive-mt

ENV PATH /opt/conda/bin:$PATH

COPY requirement.yml .

RUN apt-get update && apt-get -y install wget && \
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    conda env create -f requirement.yml && source activate && conda activate ml