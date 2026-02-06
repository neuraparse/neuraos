#!/bin/bash
#
# NeuralOS Quick Start Script
# Starts QEMU VM + Web Dashboard in one command
# Usage: ./scripts/start_neuraos.sh [arm64|x86_64]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

ARCH="${1:-x86_64}"
MEMORY="${MEMORY:-2048}"
CPUS="${CPUS:-2}"
SSH_PORT="${SSH_PORT:-2222}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8082}"
VM_API_PORT="${VM_API_PORT:-8080}"

echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║            NeuralOS Quick Start                              ║"
echo "║            Architecture: ${ARCH}                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Stop existing processes
echo -e "${YELLOW}Stopping existing processes...${NC}"
pkill -f qemu-system 2>/dev/null || true
pkill -f 'node dist/index.js' 2>/dev/null || true
sleep 2

# Determine image paths
if [ "$ARCH" = "arm64" ]; then
    KERNEL="$PROJECT_ROOT/neuraos-images/Image"
    ROOTFS="$PROJECT_ROOT/neuraos-images/rootfs.ext2"
    QEMU_CMD="qemu-system-aarch64"
    QEMU_ARGS="-machine virt -cpu cortex-a57"
    CONSOLE="ttyAMA0"
else
    KERNEL="$PROJECT_ROOT/neuraos-images-x86_64/bzImage"
    ROOTFS="$PROJECT_ROOT/neuraos-images-x86_64/rootfs.ext2"
    QEMU_CMD="qemu-system-x86_64"
    # Use KVM if available
    if [ -e /dev/kvm ]; then
        QEMU_ARGS="-enable-kvm -cpu host"
        echo -e "${GREEN}KVM acceleration enabled${NC}"
    else
        QEMU_ARGS="-cpu max"
        echo -e "${YELLOW}KVM not available, using software emulation${NC}"
    fi
    CONSOLE="ttyS0"
fi

# Check images exist
if [ ! -f "$KERNEL" ]; then
    echo -e "${RED}Kernel image not found: $KERNEL${NC}"
    echo "Please build NeuralOS first: make all"
    exit 1
fi

if [ ! -f "$ROOTFS" ]; then
    echo -e "${RED}Root filesystem not found: $ROOTFS${NC}"
    exit 1
fi

# Start QEMU VM
echo -e "${CYAN}Starting NeuralOS VM...${NC}"
if [ "$ARCH" = "arm64" ]; then
    $QEMU_CMD \
        $QEMU_ARGS \
        -m "$MEMORY" -smp "$CPUS" \
        -kernel "$KERNEL" \
        -drive if=none,file="$ROOTFS",id=rootdisk,format=raw \
        -device virtio-blk-device,drive=rootdisk \
        -append "root=/dev/vda rw console=$CONSOLE init=/sbin/init loglevel=3" \
        -device virtio-net-device,netdev=net0 \
        -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22,hostfwd=tcp::${VM_API_PORT}-:8080 \
        -display none \
        -serial file:/tmp/qemu_neuraos_serial.log \
        -monitor none \
        -daemonize \
        -pidfile /tmp/qemu_neuraos.pid 2>/dev/null || {
        # Fallback for older QEMU without -daemonize + -display none
        nohup $QEMU_CMD \
            $QEMU_ARGS \
            -m "$MEMORY" -smp "$CPUS" \
            -kernel "$KERNEL" \
            -drive if=none,file="$ROOTFS",id=rootdisk,format=raw \
            -device virtio-blk-device,drive=rootdisk \
            -append "root=/dev/vda rw console=$CONSOLE init=/sbin/init loglevel=3" \
            -device virtio-net-device,netdev=net0 \
            -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22,hostfwd=tcp::${VM_API_PORT}-:8080 \
            -nographic -no-reboot > /tmp/qemu_neuraos_serial.log 2>&1 &
        echo $! > /tmp/qemu_neuraos.pid
    }
else
    $QEMU_CMD \
        $QEMU_ARGS \
        -m "$MEMORY" -smp "$CPUS" \
        -kernel "$KERNEL" \
        -drive file="$ROOTFS",format=raw,if=virtio \
        -append "root=/dev/vda rw console=$CONSOLE init=/sbin/init loglevel=3" \
        -device virtio-net-pci,netdev=net0 \
        -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22,hostfwd=tcp::${VM_API_PORT}-:8080 \
        -display none \
        -serial file:/tmp/qemu_neuraos_serial.log \
        -monitor none \
        -daemonize \
        -pidfile /tmp/qemu_neuraos.pid
fi

QEMU_PID=$(cat /tmp/qemu_neuraos.pid 2>/dev/null)
echo -e "${GREEN}VM started (PID: $QEMU_PID)${NC}"

# Wait for boot
echo -e "${CYAN}Waiting for VM to boot...${NC}"
for i in $(seq 1 30); do
    if sshpass -p neuraos ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=2 -p "$SSH_PORT" root@localhost 'true' 2>/dev/null; then
        echo -e "${GREEN}VM is ready!${NC}"
        break
    fi
    printf "."
    sleep 1
done
echo ""

# Start Web Dashboard (if built)
DASHBOARD_DIR="$PROJECT_ROOT/web/backend"
if [ -f "$DASHBOARD_DIR/dist/index.js" ]; then
    echo -e "${CYAN}Starting Web Dashboard...${NC}"
    cd "$DASHBOARD_DIR"
    PORT=$DASHBOARD_PORT HOST=0.0.0.0 NODE_ENV=production \
    JWT_SECRET=$(head -c 32 /dev/urandom | base64) \
    JWT_REFRESH_SECRET=$(head -c 32 /dev/urandom | base64) \
    DB_PATH=/tmp/neuraos-dashboard.db \
    DEFAULT_ADMIN_PASSWORD=admin123 \
    nohup node dist/index.js > /tmp/neuraos_dashboard.log 2>&1 &
    echo $! > /tmp/neuraos_dashboard.pid
    sleep 2
    echo -e "${GREEN}Dashboard started (PID: $(cat /tmp/neuraos_dashboard.pid))${NC}"
else
    echo -e "${YELLOW}Dashboard not built. Run: cd web && pnpm install && pnpm -r build${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║            NeuralOS is Running!                               ║${NC}"
echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Access Points:${NC}"
echo -e "  ${CYAN}Web Dashboard:${NC}  http://localhost:${DASHBOARD_PORT}  (admin / admin123)"
echo -e "  ${CYAN}SSH:${NC}            ssh -p ${SSH_PORT} root@localhost  (password: neuraos)"
echo -e "  ${CYAN}VM API:${NC}         http://localhost:${VM_API_PORT}/api/metrics"
echo ""
echo -e "${BOLD}Quick Commands:${NC}"
echo -e "  ${CYAN}Stop VM:${NC}        kill $(cat /tmp/qemu_neuraos.pid 2>/dev/null)"
echo -e "  ${CYAN}Stop Dashboard:${NC} kill $(cat /tmp/neuraos_dashboard.pid 2>/dev/null)"
echo -e "  ${CYAN}Stop All:${NC}       ./scripts/stop_neuraos.sh"
echo -e "  ${CYAN}VM Serial Log:${NC}  tail -f /tmp/qemu_neuraos_serial.log"
echo ""
