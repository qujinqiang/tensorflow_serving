FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV TF_CUDA_VERSION=9.0 
TF_CUDNN_VERSION=7 
TF_SERVING_BRANCH=r1.7 
BAZEL_VERSION=0.10.0

RUN apt-get update && apt-get install -y 
build-essential 
curl 
git 
libfreetype6-dev 
libpng12-dev 
libzmq3-dev 
mlocate 
pkg-config 
python-dev 
python-numpy 
python-pip 
software-properties-common 
swig 
zip 
zlib1g-dev 
libcurl3-dev 
openjdk-8-jdk
openjdk-8-jre-headless 
wget 
&& 
apt-get clean && 
rm -rf /var/lib/apt/lists/*

RUN pip install mock grpcio

ENV BAZELRC /root/.bazelrc

WORKDIR /root/
RUN mkdir /bazel && 
cd /bazel && 
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && 
curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && 
chmod +x bazel-*.sh && 
./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && 
cd / && 
rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

ENV TF_NEED_CUDA=1 
TF_NEED_S3=1 
TF_CUDA_COMPUTE_CAPABILITIES="3.7" 
TF_NEED_GCP=1 
TF_NEED_JEMALLOC=0 
TF_NEED_HDFS=0 
TF_NEED_OPENCL=0 
TF_NEED_MKL=0 
TF_NEED_VERBS=0 
TF_NEED_MPI=0 
TF_NEED_GDR=0 
TF_ENABLE_XLA=0 
TF_CUDA_CLANG=0 
TF_NEED_OPENCL_SYCL=0 
CUDA_TOOLKIT_PATH=/usr/local/cuda 
CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu 
GCC_HOST_COMPILER_PATH=/usr/bin/gcc 
PYTHON_BIN_PATH=/usr/bin/python 
CC_OPT_FLAGS="-march=native" 
PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages

RUN git clone -b r1.6 https://github.com/tensorflow/serving.git

COPY parameters /root/parameters

RUN cd /root/serving && 
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --verbose_failures 
--crosstool_top=@local_config_cuda//crosstool:toolchain --spawn_strategy=standalone tensorflow_serving/model_servers:tensorflow_model_server

RUN ln -s /usr/local/cuda /usr/local/nvidia && 
ln -s /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1

WORKDIR /root/serving

CMD bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server 
--enable_batching 
--batching_parameters_file=/root/parameters/parameters.proto 
--model_config_file=/root/parameters/model_config
