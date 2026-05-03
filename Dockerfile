FROM nvcr.io/nvidia/l4t-base:r36.2.0

WORKDIR /workspace

# Install build tools (nvcc is already available via host runtime)
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]