
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    zip \
    unzip \
    byobu \
    vim \
    nano

WORKDIR /workspace/rsr

# RUN chmod -R a+w .
COPY requirements.txt /workspace/rsr
RUN pip install -r requirements.txt --ignore-installed

