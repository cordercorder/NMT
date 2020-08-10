FROM nvidia/cuda:9.0-devel-ubuntu16.04

WORKDIR /nmt

ENV PATH /opt/conda/bin:$PATH

COPY *requirement.txt ./

RUN apt-get update && apt-get -y install wget && \
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/simpleitk/ && \
    conda config --add channels pytorch && \
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free && \
    conda config --set show_channel_urls yes && \
    conda config --set remote_connect_timeout_secs 64 && \
    conda config --set remote_read_timeout_secs 128 && \
    conda config --set remote_max_retries 5