FROM --platform=linux/amd64 python:3.12

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gdb \
    wget \
    unzip

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch torchvision

CMD ["/bin/bash"]