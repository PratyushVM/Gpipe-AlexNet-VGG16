# Run the following commands in order:
#
# LINGVO_DIR="/tmp/lingvo"  # (change to the cloned lingvo directory, e.g. "$HOME/lingvo")
# LINGVO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
# docker build --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04") - < "$LINGVO_DIR/docker/dev.dockerfile"
#
# previous command gives cudnn8.1.1 
#
# docker run $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash
# 
# Test that everything worked:
#
# bazel test -c opt --test_output=streamed //lingvo:trainer_test //lingvo:models_test
#
# To start image 
#
# docker start -i lingvo 


ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Lingvo team <lingvo-bot@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends software-properties-common
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  aria2 \
  build-essential \
  curl \
  dirmngr \
  git \
  gpg-agent \
  less \
  libfreetype6-dev \
  libhdf5-serial-dev \
  libpng-dev \
  libzmq3-dev \
  lsof \
  pkg-config \
  rename \
  rsync \
  sox \
  unzip \
  vim \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install python 3.8
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" > /etc/apt/sources.list.d/deadsnakes-ppa-bionic.list
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1000
# bazel assumes the python executable is "python".
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1000

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

ARG bazel_version=4.0.0
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
  cd /bazel && \
  curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
  curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
  chmod +x bazel-*.sh && \
  ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
  cd / && \
  rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# TODO(austinwaters): Remove graph-compression-google-research once
# model-pruning-google-research properly declares it as a dependency.
ARG pip_dependencies=' \
  apache-beam[gcp]>=2.8 \
  attrs \
  contextlib2 \
  dataclasses \
  google-api-python-client \
  graph-compression-google-research \
  h5py \
  ipykernel \
  jupyter \
  jupyter_http_over_ws \
  matplotlib \
  mock \
  model-pruning-google-research \
  numpy \
  oauth2client \
  pandas \
  Pillow \
  pyyaml \
  recommonmark \
  scikit-learn \
  scipy \
  sphinx \
  sphinx_rtd_theme \
  sympy'

RUN pip3 --no-cache-dir install $pip_dependencies
RUN python3 -m ipykernel.kernelspec

# The latest tensorflow requires CUDA 10 compatible nvidia drivers (410.xx).
# If you are unable to update your drivers, an alternative is to compile
# tensorflow from source instead of installing from pip.
# Ensure we install the correct version by uninstalling first.
RUN pip3 uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu
RUN pip3 --no-cache-dir install tensorflow tensorflow-datasets \
  tensorflow-hub tensorflow-text

RUN jupyter serverextension enable --py jupyter_http_over_ws

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

WORKDIR "/tmp/lingvo"

CMD ["/bin/bash"]
