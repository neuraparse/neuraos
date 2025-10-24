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
    libpthread-stubs0-dev \
    linux-libc-dev \
    cpio \
    qemu-system-arm \
    qemu-system-aarch64 \
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

# Create minimal rootfs
RUN ./scripts/create_minimal_rootfs.sh

# Default command: run tests
CMD ["sh", "-c", "cd build && ctest --output-on-failure --verbose"]

