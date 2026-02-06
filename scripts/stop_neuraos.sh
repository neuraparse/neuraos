#!/bin/bash
#
# NeuralOS Stop Script
# Stops QEMU VM + Web Dashboard
#

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Stopping NeuralOS services..."

# Stop Dashboard
if [ -f /tmp/neuraos_dashboard.pid ]; then
    PID=$(cat /tmp/neuraos_dashboard.pid)
    kill "$PID" 2>/dev/null && echo -e "${GREEN}Dashboard stopped (PID: $PID)${NC}" || echo -e "${YELLOW}Dashboard was not running${NC}"
    rm -f /tmp/neuraos_dashboard.pid
fi

# Stop QEMU
if [ -f /tmp/qemu_neuraos.pid ]; then
    PID=$(cat /tmp/qemu_neuraos.pid)
    kill "$PID" 2>/dev/null && echo -e "${GREEN}VM stopped (PID: $PID)${NC}" || echo -e "${YELLOW}VM was not running${NC}"
    rm -f /tmp/qemu_neuraos.pid
fi

# Cleanup any remaining processes
pkill -f 'node dist/index.js' 2>/dev/null || true
pkill -f qemu-system 2>/dev/null || true

echo -e "${GREEN}All NeuralOS services stopped.${NC}"
