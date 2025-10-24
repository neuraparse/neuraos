#!/bin/bash
#
# NeuralOS Build Script
# Builds a complete bootable NeuralOS image using Buildroot
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILDROOT_VERSION="2024.02.9"
BUILDROOT_DIR="$PROJECT_ROOT/buildroot-$BUILDROOT_VERSION"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."
    
    local missing_deps=()
    
    for cmd in wget tar make gcc g++ patch perl python3 rsync bc cpio unzip; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "On Ubuntu/Debian: sudo apt-get install build-essential wget cpio unzip rsync bc"
        log_info "On macOS: brew install wget coreutils"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Download Buildroot
download_buildroot() {
    if [ -d "$BUILDROOT_DIR" ]; then
        log_info "Buildroot already downloaded: $BUILDROOT_DIR"
        return 0
    fi
    
    log_info "Downloading Buildroot $BUILDROOT_VERSION..."
    
    cd "$PROJECT_ROOT"
    wget -c "https://buildroot.org/downloads/buildroot-$BUILDROOT_VERSION.tar.gz"
    tar xzf "buildroot-$BUILDROOT_VERSION.tar.gz"
    rm "buildroot-$BUILDROOT_VERSION.tar.gz"
    
    log_success "Buildroot downloaded"
}

# Configure Buildroot
configure_buildroot() {
    log_info "Configuring Buildroot with NeuralOS defconfig..."

    cd "$BUILDROOT_DIR"

    # Load our defconfig with BR2_EXTERNAL
    # Use minimal config (no GUI) to avoid legacy package issues
    make BR2_EXTERNAL="$PROJECT_ROOT" neuraos_minimal_defconfig

    log_success "Buildroot configured"
}

# Build NeuralOS
build_neuraos() {
    log_info "Building NeuralOS (this will take a while - 30min to 2 hours)..."
    log_info "Build directory: $BUILDROOT_DIR"

    cd "$BUILDROOT_DIR"

    # Build with all CPU cores
    local ncpus=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    log_info "Building with $ncpus parallel jobs..."

    make BR2_EXTERNAL="$PROJECT_ROOT" -j$ncpus

    log_success "Build completed!"
}

# Show build results
show_results() {
    log_info "Build artifacts:"
    echo ""
    
    local images_dir="$BUILDROOT_DIR/output/images"
    
    if [ -d "$images_dir" ]; then
        ls -lh "$images_dir"
        echo ""
        log_success "Images available at: $images_dir"
        echo ""
        log_info "To test with QEMU:"
        echo "  cd $BUILDROOT_DIR"
        echo "  qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \\"
        echo "    -kernel output/images/Image \\"
        echo "    -append 'console=ttyAMA0' \\"
        echo "    -nographic"
    else
        log_error "Build artifacts not found!"
        exit 1
    fi
}

# Main
main() {
    log_info "NeuralOS Build Script"
    log_info "====================="
    echo ""
    
    check_dependencies
    download_buildroot
    configure_buildroot
    build_neuraos
    show_results
    
    log_success "All done!"
}

# Parse arguments
case "${1:-}" in
    clean)
        log_info "Cleaning build artifacts..."
        rm -rf "$BUILDROOT_DIR"
        log_success "Clean complete"
        ;;
    config)
        check_dependencies
        download_buildroot
        configure_buildroot
        log_info "Run 'make menuconfig' in $BUILDROOT_DIR to customize"
        ;;
    *)
        main
        ;;
esac

