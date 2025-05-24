# Use an official NVIDIA base image with CUDA 11.6 and Ubuntu 22.04
# Run with sudo docker run --gpus all -it --rm -v $(pwd):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:2.12.0-gpu-jupyter
FROM nvidia/cuda:11.6.1-base-ubuntu20.04
# Set environment variables
ENV LANG C.UTF-8


# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev

# Update pip
RUN pip install --upgrade pip

# Install cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev

# Install TensorFlow 2.12
RUN pip install tensorflow==2.12.0
RUN pip install jupyter
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install akida==2.7.2
RUN pip install cnn2snn==2.7.2
RUN pip install imblearn
RUN pip install akida-models==1.5.0


COPY . /workspace

EXPOSE 8888


WORKDIR /workspace

# Set the default command to python3
CMD ["bash"]
