#!/bin/bash
# NeuralOS System Test Suite
# Tests all critical components of the system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGES_DIR="$PROJECT_ROOT/buildroot-images-pci"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test functions
test_qemu_installed() {
    log_info "Testing QEMU installation..."
    if command -v qemu-system-aarch64 &> /dev/null; then
        QEMU_VERSION=$(qemu-system-aarch64 --version | head -1)
        log_success "QEMU installed: $QEMU_VERSION"
        return 0
    else
        log_error "QEMU not installed"
        return 1
    fi
}

test_images_exist() {
    log_info "Testing build artifacts..."
    
    if [ ! -d "$IMAGES_DIR" ]; then
        log_error "Images directory not found: $IMAGES_DIR"
        return 1
    fi
    
    local all_exist=true
    
    if [ -f "$IMAGES_DIR/Image" ]; then
        SIZE=$(ls -lh "$IMAGES_DIR/Image" | awk '{print $5}')
        log_success "Kernel image exists: $SIZE"
    else
        log_error "Kernel image not found"
        all_exist=false
    fi
    
    if [ -f "$IMAGES_DIR/rootfs.ext4" ]; then
        SIZE=$(ls -lh "$IMAGES_DIR/rootfs.ext4" | awk '{print $5}')
        log_success "Root filesystem exists: $SIZE"
    else
        log_error "Root filesystem not found"
        all_exist=false
    fi
    
    if [ -f "$IMAGES_DIR/u-boot.bin" ]; then
        SIZE=$(ls -lh "$IMAGES_DIR/u-boot.bin" | awk '{print $5}')
        log_success "U-Boot bootloader exists: $SIZE"
    else
        log_error "U-Boot bootloader not found"
        all_exist=false
    fi
    
    [ "$all_exist" = true ]
}

test_kernel_config() {
    log_info "Testing kernel configuration..."
    
    local config_file="$PROJECT_ROOT/configs/kernel/neuraos_defconfig"
    
    if [ ! -f "$config_file" ]; then
        log_error "Kernel config not found: $config_file"
        return 1
    fi
    
    local all_ok=true
    
    # Check critical configs
    if grep -q "CONFIG_PCI=y" "$config_file"; then
        log_success "PCI support enabled"
    else
        log_error "PCI support not enabled"
        all_ok=false
    fi
    
    if grep -q "CONFIG_VIRTIO_BLK=y" "$config_file"; then
        log_success "VIRTIO block driver enabled"
    else
        log_error "VIRTIO block driver not enabled"
        all_ok=false
    fi
    
    if grep -q "CONFIG_VIRTIO_NET=y" "$config_file"; then
        log_success "VIRTIO network driver enabled"
    else
        log_error "VIRTIO network driver not enabled"
        all_ok=false
    fi
    
    if grep -q "CONFIG_VIRTIO_PCI=y" "$config_file"; then
        log_success "VIRTIO PCI transport enabled"
    else
        log_error "VIRTIO PCI transport not enabled"
        all_ok=false
    fi
    
    [ "$all_ok" = true ]
}

test_boot_quick() {
    log_info "Testing quick boot (10 seconds)..."
    
    cd "$IMAGES_DIR"
    
    # Boot for 10 seconds and capture output
    timeout 10 qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
        -kernel Image \
        -drive file=rootfs.ext4,if=virtio,format=raw \
        -append "root=/dev/vda console=ttyAMA0" \
        -nographic 2>&1 | tee /tmp/neuraos_quick_boot.log || true
    
    # Check for critical boot messages
    if grep -q "Linux version 6.12.8-neuraos" /tmp/neuraos_quick_boot.log; then
        log_success "Kernel version detected"
    else
        log_error "Kernel version not found in boot log"
        return 1
    fi
    
    if grep -q "virtio_blk virtio1: \[vda\]" /tmp/neuraos_quick_boot.log; then
        log_success "VIRTIO block device detected"
    else
        log_error "VIRTIO block device not detected"
        return 1
    fi
    
    if grep -q "EXT4-fs (vda): mounted filesystem" /tmp/neuraos_quick_boot.log; then
        log_success "Root filesystem mounted"
    else
        log_error "Root filesystem not mounted"
        return 1
    fi
    
    if grep -q "Starting network:" /tmp/neuraos_quick_boot.log; then
        log_success "Network service started"
    else
        log_error "Network service not started"
        return 1
    fi
    
    if grep -q "Starting dropbear sshd:" /tmp/neuraos_quick_boot.log; then
        log_success "SSH server started"
    else
        log_error "SSH server not started"
        return 1
    fi
    
    if grep -q "Welcome to NeuralOS" /tmp/neuraos_quick_boot.log; then
        log_success "Login prompt displayed"
    else
        log_error "Login prompt not displayed"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
}

test_buildroot_config() {
    log_info "Testing Buildroot configuration..."
    
    local config_file="$PROJECT_ROOT/configs/neuraos_minimal_defconfig"
    
    if [ ! -f "$config_file" ]; then
        log_error "Buildroot config not found: $config_file"
        return 1
    fi
    
    if grep -q "BR2_aarch64=y" "$config_file"; then
        log_success "ARM64 architecture configured"
    else
        log_error "ARM64 architecture not configured"
        return 1
    fi
    
    if grep -q "BR2_LINUX_KERNEL=y" "$config_file"; then
        log_success "Linux kernel enabled"
    else
        log_error "Linux kernel not enabled"
        return 1
    fi
    
    if grep -q "BR2_TARGET_ROOTFS_EXT2=y" "$config_file"; then
        log_success "EXT4 rootfs enabled"
    else
        log_error "EXT4 rootfs not enabled"
        return 1
    fi
}

test_docker_setup() {
    log_info "Testing Docker setup..."
    
    if [ ! -f "$PROJECT_ROOT/Dockerfile.buildroot" ]; then
        log_error "Dockerfile not found"
        return 1
    fi
    
    log_success "Dockerfile exists"
    
    if command -v docker &> /dev/null; then
        log_success "Docker installed"
    else
        log_warning "Docker not installed (optional for macOS users)"
    fi
}

# Print header
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              NEURAOS SYSTEM TEST SUITE                        ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Run tests
test_qemu_installed
test_images_exist
test_kernel_config
test_buildroot_config
test_docker_setup
test_boot_quick

# Print summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED!${NC}"
    echo ""
    exit 1
fi

