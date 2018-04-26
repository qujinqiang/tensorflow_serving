# docker build --pull -t tf/tensorflow-serving --label 1.6 -f Dockerfile .
# export TF_SERVING_PORT=9000
# export TF_SERVING_MODEL_PATH=/tf_models/mymodel
# export CONTAINER_NAME=tf_serving_1_6
# CUDA_VISIBLE_DEVICES=0 docker run --runtime=nvidia -it -p $TF_SERVING_PORT:$TF_SERVING_PORT -v $TF_SERVING_MODEL_PATH:/root/tf_model --name $CONTAINER_NAME tf/tensorflow-serving /usr/local/bin/tensorflow_model_server --port=$TF_SERVING_PORT --enable_batching=true --model_base_path=/root/tf_model/
# docker start -ai $CONTAINER_NAME

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


# CUDA and CUDNN versions (must match the image source)

ENV TF_CUDA_VERSION=9.0 \
	TF_CUDNN_VERSION=7 \
	TF_SERVING_COMMIT=tags/1.6.0 \
	BAZEL_VERSION=0.11.1


# Set up ubuntu packages

RUN apt-get update && apt-get install -y \
		build-essential \
		curl \
		git \
		libfreetype6-dev \
		libpng12-dev \
		libzmq3-dev \
		mlocate \
		pkg-config \
		python-dev \
		python-numpy \
		python-pip \
		software-properties-common \
		swig \
		zip \
		zlib1g-dev \
		libcurl3-dev \
		openjdk-8-jdk\
		openjdk-8-jre-headless \
		wget \
		&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*


# Set up grpc

RUN pip install mock grpcio


# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf: https://github.com/bazelbuild/bazel/issues/134
RUN echo "startup --batch" >>/root/.bazelrc
# Similarly, we need to workaround sandboxing issues: https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/root/.bazelrc
ENV BAZELRC /root/.bazelrc


# Install the most recent bazel release.

WORKDIR /bazel
RUN curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
	chmod +x bazel-*.sh && \
	./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# Fix paths so that CUDNN can be found: https://github.com/tensorflow/tensorflow/issues/8264

WORKDIR /
RUN mkdir /usr/lib/x86_64-linux-gnu/include/ && \
	ln -s /usr/lib/x86_64-linux-gnu/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h && \
	ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h && \
	ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
	ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.$TF_CUDNN_VERSION /usr/local/cuda/lib64/libcudnn.so.$TF_CUDNN_VERSION


# Enable CUDA support

ENV TF_NEED_CUDA=1 \
	TF_CUDA_COMPUTE_CAPABILITIES="3.0,3.5,5.2,6.0,6.1" \
	LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Download TensorFlow Serving

WORKDIR /tensorflow
RUN git clone --recurse-submodules https://github.com/tensorflow/serving


# Build TensorFlow Serving

WORKDIR /tensorflow/serving
RUN bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --verbose_failures --crosstool_top=@local_config_cuda//crosstool:toolchain tensorflow_serving/model_servers:tensorflow_model_server


# Install tensorflow_model_server and clean bazel

RUN cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
	bazel clean --expunge


CMD ["/bin/bash"]
