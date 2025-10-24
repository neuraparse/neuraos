#!/bin/sh
#
# Start Weston compositor and NeuralOS AI Dashboard
#

export XDG_RUNTIME_DIR=/run/user/0
mkdir -p $XDG_RUNTIME_DIR
chmod 0700 $XDG_RUNTIME_DIR

# Set environment for Wayland
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
export QT_WAYLAND_DISABLE_WINDOWDECORATION=1

# Start Weston
echo "Starting Weston compositor..."
weston --backend=drm-backend.so --tty=1 --log=/var/log/weston.log &
WESTON_PID=$!

# Wait for Weston to start
sleep 2

# Check if Weston is running
if ! kill -0 $WESTON_PID 2>/dev/null; then
    echo "Weston failed to start, trying fbdev backend..."
    weston --backend=fbdev-backend.so --tty=1 --log=/var/log/weston.log &
    WESTON_PID=$!
    sleep 2
fi

# Start AI Dashboard
if [ -x /usr/bin/neuraos-dashboard ]; then
    echo "Starting NeuralOS AI Dashboard..."
    sleep 1
    /usr/bin/neuraos-dashboard &
else
    echo "AI Dashboard not found, starting weston-terminal..."
    weston-terminal &
fi

# Keep script running
wait

