FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    clang-format \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /neuraos

# Copy source code
COPY . .

# Build the project
RUN rm -rf build && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DNEURAOS_ENABLE_LITERT=OFF \
        -DNEURAOS_ENABLE_ONNX=OFF \
        -DNEURAOS_ENABLE_WASMEDGE=OFF && \
    cmake --build build -j$(nproc)

# Default command: run tests
CMD ["sh", "-c", "cd build && ctest --output-on-failure --verbose"]

