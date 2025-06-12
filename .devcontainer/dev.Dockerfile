FROM mcr.microsoft.com/devcontainers/base:jammy
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

ARG DEBIAN_FRONT_END=noninteractive
ARG TZ=Etc/UTC

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        build-essential \
        tmux \
        tree \
        vim \
        xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
