#!/bin/bash
#
# NeuralOS x86_64 Build Script
# Builds a KVM-ready x86_64 image for production servers
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILDROOT_VERSION="2024.02.9"
BUILDROOT_DIR="$PROJECT_ROOT/buildroot-x86_64-$BUILDROOT_VERSION"
OUTPUT_DIR="$PROJECT_ROOT/neuraos-images-x86_64"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dependencies() {
    log_info "Checking build dependencies..."
    for cmd in wget tar make gcc g++ patch perl python3 rsync bc cpio unzip; do
        if ! command -v $cmd &> /dev/null; then
            log_error "Missing: $cmd"
            exit 1
        fi
    done
    log_success "All dependencies found"
}

download_buildroot() {
    if [ -d "$BUILDROOT_DIR" ]; then
        log_info "Buildroot already exists: $BUILDROOT_DIR"
        return 0
    fi

    log_info "Downloading Buildroot $BUILDROOT_VERSION..."
    cd "$PROJECT_ROOT"
    wget -c "https://buildroot.org/downloads/buildroot-$BUILDROOT_VERSION.tar.gz" -O buildroot-x86_64.tar.gz
    mkdir -p "$BUILDROOT_DIR"
    tar xzf buildroot-x86_64.tar.gz --strip-components=1 -C "$BUILDROOT_DIR"
    rm buildroot-x86_64.tar.gz
    log_success "Buildroot downloaded"
}

configure_buildroot() {
    log_info "Configuring Buildroot for x86_64..."
    cd "$BUILDROOT_DIR"
    make BR2_EXTERNAL="$PROJECT_ROOT" neuraos_x86_64_defconfig
    log_success "Configuration complete"
}

build_neuraos() {
    log_info "Building NeuralOS x86_64 (this takes 30-60 minutes)..."
    cd "$BUILDROOT_DIR"

    local ncpus=$(nproc 2>/dev/null || echo 4)
    log_info "Using $ncpus parallel jobs..."

    make BR2_EXTERNAL="$PROJECT_ROOT" -j$ncpus
    log_success "Build completed!"
}

copy_images() {
    log_info "Copying images to $OUTPUT_DIR..."
    mkdir -p "$OUTPUT_DIR"

    cp "$BUILDROOT_DIR/output/images/bzImage" "$OUTPUT_DIR/" 2>/dev/null || \
    cp "$BUILDROOT_DIR/output/images/vmlinux" "$OUTPUT_DIR/" 2>/dev/null || true

    cp "$BUILDROOT_DIR/output/images/rootfs.ext2" "$OUTPUT_DIR/" 2>/dev/null || \
    cp "$BUILDROOT_DIR/output/images/rootfs.ext4" "$OUTPUT_DIR/" 2>/dev/null || true

    ls -lh "$OUTPUT_DIR/"
    log_success "Images ready at: $OUTPUT_DIR"
}

main() {
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          NeuralOS x86_64 Build for KVM                       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    check_dependencies
    download_buildroot
    configure_buildroot
    build_neuraos
    copy_images

    echo ""
    log_success "Build complete! Run with:"
    echo "  ./scripts/run_qemu_kvm.sh"
}

case "${1:-}" in
    clean)
        log_info "Cleaning x86_64 build..."
        rm -rf "$BUILDROOT_DIR" "$OUTPUT_DIR"
        log_success "Clean complete"
        ;;
    *)
        main
        ;;
esac
