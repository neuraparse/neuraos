#!/bin/bash
# Download ONNX Runtime external dependencies for v1.23.2
# This script bypasses CMake's FetchContent which requires HTTPS-enabled libcurl
# Version info from: https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/deps.txt

set -e

DEPS_DIR="$1"
if [ -z "$DEPS_DIR" ]; then
    echo "Usage: $0 <deps_directory>"
    exit 1
fi

mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

download_and_extract() {
    local name="$1"
    local url="$2"
    local extracted_name="$3"
    local target_name="$4"

    echo "Downloading $name..."
    if wget -q -O "${name}.zip" "$url"; then
        unzip -q -o "${name}.zip"
        if [ -d "$extracted_name" ]; then
            mv "$extracted_name" "$target_name"
            rm -f "${name}.zip"
            echo "  -> $target_name ready"
        else
            echo "  ERROR: Expected directory '$extracted_name' not found after extraction"
            ls -la
            exit 1
        fi
    else
        echo "  ERROR: Failed to download $url"
        exit 1
    fi
}

echo "=== Downloading ONNX Runtime dependencies (v1.23.2) ==="

# abseil-cpp (from deps.txt)
download_and_extract "abseil-cpp" \
    "https://github.com/abseil/abseil-cpp/archive/refs/tags/20250512.0.zip" \
    "abseil-cpp-20250512.0" \
    "abseil_cpp-src"

# eigen (from deps.txt)
download_and_extract "eigen" \
    "https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip" \
    "eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33" \
    "eigen-src"

# cpuinfo (from deps.txt)
download_and_extract "cpuinfo" \
    "https://github.com/pytorch/cpuinfo/archive/8a1772a0c5c447df2d18edf33ec4603a8c9c04a6.zip" \
    "cpuinfo-8a1772a0c5c447df2d18edf33ec4603a8c9c04a6" \
    "cpuinfo-src"

# onnx (from deps.txt)
download_and_extract "onnx" \
    "https://github.com/onnx/onnx/archive/refs/tags/v1.18.0.zip" \
    "onnx-1.18.0" \
    "onnx-src"

# safeint (from deps.txt)
download_and_extract "safeint" \
    "https://github.com/dcleblanc/SafeInt/archive/refs/tags/3.0.28.zip" \
    "SafeInt-3.0.28" \
    "safeint-src"

# microsoft GSL (from deps.txt) - FetchContent name is "GSL"
download_and_extract "gsl" \
    "https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip" \
    "GSL-4.0.0" \
    "gsl-src"

# nlohmann json (from deps.txt)
download_and_extract "json" \
    "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip" \
    "json-3.11.3" \
    "nlohmann_json-src"

# date (from deps.txt)
download_and_extract "date" \
    "https://github.com/HowardHinnant/date/archive/refs/tags/v3.0.1.zip" \
    "date-3.0.1" \
    "date-src"

# mp11 / boost (from deps.txt)
download_and_extract "mp11" \
    "https://github.com/boostorg/mp11/archive/refs/tags/boost-1.82.0.zip" \
    "mp11-boost-1.82.0" \
    "mp11-src"

# re2 (from deps.txt)
download_and_extract "re2" \
    "https://github.com/google/re2/archive/refs/tags/2024-07-02.zip" \
    "re2-2024-07-02" \
    "re2-src"

# flatbuffers (from deps.txt)
download_and_extract "flatbuffers" \
    "https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.zip" \
    "flatbuffers-23.5.26" \
    "flatbuffers-src"

# protobuf (from deps.txt) - needed for ONNX
download_and_extract "protobuf" \
    "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.12.zip" \
    "protobuf-21.12" \
    "protobuf-src"

echo "=== All dependencies downloaded successfully ==="
