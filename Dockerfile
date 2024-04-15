# Dockerfile for CandidateVectorSearch Linux/Ubuntu Build
# author: Micha Birklbauer
# version: 1.0.0

FROM ubuntu:22.04

LABEL maintainer="micha.birklbauer@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gnupg \
    gnupg1 \
    gnupg2 \
    software-properties-common \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libc6 \
    libgcc-s1 \
    libgssapi-krb5-2 \
    libicu70 \
    liblttng-ust1 \
    libssl3 \
    libstdc++6 \
    libunwind8 \
    zlib1g \
    dotnet-sdk-6.0 \
    dotnet-runtime-6.0 \
    unzip \
    gcc \
    g++\
    nano

RUN mkdir CandidateVectorSearch
COPY ./ CandidateVectorSearch/
WORKDIR CandidateVectorSearch

CMD  ["/bin/bash"]
