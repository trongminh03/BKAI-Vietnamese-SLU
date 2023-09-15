FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

WORKDIR /SLU

# install dependencies
RUN --mount=type=cache,target=/root/.cache \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libjpeg-dev \
    libssl-dev \
    libpng-dev \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \ 
    libsqlite3-dev \
    libncurses5-dev \
    liblzma-dev \
    libxml2-dev \
    libxmlsec1-dev \ 
    libffi-dev \ 
    tk-dev \ 
    libgl1 \
    llvm \
    xz-utils \ 
    ccache \
    cmake \
    make \
    curl \
    mecab-ipadic-utf8 \ 
    git \
    default-jdk \
    wget \
    gcc \
    build-essential \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y

# Set-up necessary Env vars for PyEnv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# RUN --mount=type=cache,target=/root/.cache \
#     apt-get update -y \
#     && apt-get install -y python3.9 python3.9-dev \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 

# RUN --mount=type=cache,target=/root/.cache \
#     apt-get update -y \
#     && apt-get install -y python3-pip \
#     && pip3 install --upgrade pip \
#     && pip3 install pyem empy pyyaml

COPY . /SLU

# Install pyenv
RUN --mount=type=cache,target=/root/.cache \ 
    set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install 3.6 \
    && pyenv install 3.9 \
    && pyenv global 3.9 \
    && pyenv rehash

# install requirements
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt 

ARG INSTALL_TORCH="pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"

RUN --mount=type=cache,target=/root/.cache \
    bash -c "${INSTALL_TORCH}"

RUN --mount=type=cache,target=/root/.cache \
    pip install protobuf==3.20.* \ 
    && pip install jiwer \
    && pip install https://github.com/kpu/kenlm/archive/master.zip


# download pretrain weight
# RUN --mount=type=cache,target=/root/.cache \
#     pip install gdown && \
#     cd /SLU && \
#     gdown --folder 1CK__VZzxKA9ZZFbl0frFYyL3dFxEm83_ && \
#     gdown --folder 1HVt09CDAyoSaEdzC7J38VH_ksE4SgO_J

# set up kenlm
RUN --mount=type=cache,target=/root/.cache \
    pyenv global 3.6 \
    && pip install nltk 

RUN cd /SLU/kenlm \ 
    && ./bjam || true
    
RUN cd /SLU/kenlm \
    && python3 setup.py install \
    && pyenv global 3.9
