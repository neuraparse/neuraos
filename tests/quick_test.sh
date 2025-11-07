#!/bin/bash
# NeuralOS Quick Test - Fast verification without booting QEMU

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="$PROJECT_ROOT/buildroot-images-pci"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         NEURAOS QUICK VERIFICATION TEST                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: QEMU
echo -n "Testing QEMU installation... "
if command -v qemu-system-aarch64 &> /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 2: Kernel Image
echo -n "Testing kernel image... "
if [ -f "$IMAGES_DIR/Image" ]; then
    SIZE=$(ls -lh "$IMAGES_DIR/Image" | awk '{print $5}')
    echo -e "${GREEN}✅ PASS${NC} ($SIZE)"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 3: Rootfs
echo -n "Testing root filesystem... "
if [ -f "$IMAGES_DIR/rootfs.ext4" ]; then
    SIZE=$(ls -lh "$IMAGES_DIR/rootfs.ext4" | awk '{print $5}')
    echo -e "${GREEN}✅ PASS${NC} ($SIZE)"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 4: Kernel config - PCI
echo -n "Testing kernel PCI support... "
if grep -q "CONFIG_PCI=y" "$PROJECT_ROOT/configs/kernel/neuraos_defconfig"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 5: Kernel config - VIRTIO_BLK
echo -n "Testing VIRTIO block driver... "
if grep -q "CONFIG_VIRTIO_BLK=y" "$PROJECT_ROOT/configs/kernel/neuraos_defconfig"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 6: Kernel config - VIRTIO_NET
echo -n "Testing VIRTIO network driver... "
if grep -q "CONFIG_VIRTIO_NET=y" "$PROJECT_ROOT/configs/kernel/neuraos_defconfig"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 7: Kernel config - VIRTIO_PCI
echo -n "Testing VIRTIO PCI transport... "
if grep -q "CONFIG_VIRTIO_PCI=y" "$PROJECT_ROOT/configs/kernel/neuraos_defconfig"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 8: Buildroot config
echo -n "Testing Buildroot configuration... "
if [ -f "$PROJECT_ROOT/configs/neuraos_minimal_defconfig" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 9: Dockerfile
echo -n "Testing Docker build setup... "
if [ -f "$PROJECT_ROOT/Dockerfile.buildroot" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Test 10: Build script
echo -n "Testing build script... "
if [ -f "$PROJECT_ROOT/scripts/build_neuraos.sh" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((FAIL++))
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total: $((PASS + FAIL)) tests"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "System is ready to boot. Run:"
    echo ""
    echo "  cd buildroot-images-pci"
    echo "  qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \\"
    echo "    -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \\"
    echo "    -append \"root=/dev/vda console=ttyAMA0\" -nographic"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED!${NC}"
    echo ""
    exit 1
fi

