# NeuralOS GUI Build Guide

## 🎨 AI Dashboard with Wayland + Weston + Qt5

NeuralOS now includes a beautiful AI Dashboard with visual interface!

### Features

- **Wayland/Weston Compositor** - Modern display server
- **Qt5-based AI Dashboard** - Real-time inference monitoring
- **Hardware Acceleration** - Mesa3D, DRM/KMS, VirtIO GPU
- **Touch-friendly UI** - Optimized for embedded displays
- **Dark Theme** - Easy on the eyes

### AI Dashboard Features

1. **Real-time Performance Monitoring**
   - Inference latency graphs
   - CPU/Memory/NPU utilization
   - Live performance charts

2. **Model Management**
   - Load AI models
   - View model information
   - Switch between models

3. **Hardware Detection**
   - Automatic accelerator discovery
   - NPU/GPU status
   - Fallback to CPU

### Build Instructions

#### 1. Full Buildroot Build (with GUI)

```bash
# Build complete OS with GUI support
./scripts/build_neuraos.sh

# This will:
# - Download Buildroot 2024.02.9
# - Build Linux kernel 6.12.8-rt
# - Build Wayland + Weston
# - Build Qt5 framework
# - Build AI Dashboard
# - Create bootable image

# Time: 1-2 hours (depending on hardware)
```

#### 2. Test in QEMU with GUI

```bash
# After build completes, run with GUI:
./scripts/run_qemu_gui.sh

# Or headless (no display):
./scripts/run_qemu_headless.sh
```

#### 3. Quick Docker Test (no GUI)

```bash
# For quick userspace testing:
docker build -t neuraos-test .
docker run --rm neuraos-test
```

### QEMU Requirements

**macOS:**
```bash
brew install qemu
```

**Linux:**
```bash
sudo apt-get install qemu-system-arm qemu-system-aarch64
```

### Boot Sequence

1. **Kernel Boot** - Linux 6.12.8-rt with PREEMPT_RT
2. **Init System (NPI)** - Fast boot, minimal overhead
3. **Weston Compositor** - Wayland display server
4. **AI Dashboard** - Qt5 application launches automatically

### Display Output

The AI Dashboard shows:

```
╔════════════════════════════════════════════════════════════════╗
║                  🧠 NeuralOS AI Dashboard                      ║
╚════════════════════════════════════════════════════════════════╝

System: Online | NPIE: v1.0.0-alpha

┌─────────────────────┬──────────────────────────────────────────┐
│ AI Models           │ Inference Performance                    │
├─────────────────────┤                                          │
│ 📊 Image Class...   │  [Live Chart: Latency over time]        │
│ 🎯 Object Detect... │                                          │
│ 📝 Text Analysis    │  Hardware Utilization:                   │
│ 🔊 Audio Process... │  CPU:    [████████░░] 80%               │
│                     │  Memory: [██████░░░░] 60%               │
│ [Load Model]        │  NPU:    [███░░░░░░░] 30%               │
│                     │                                          │
│                     │  Detected Accelerators:                  │
│                     │  ✓ ARM Mali GPU                          │
│                     │  ✓ CPU fallback available               │
└─────────────────────┴──────────────────────────────────────────┘

NeuralOS v1.0.0-alpha | AI-Native Embedded Operating System
```

### Configuration Files

- **Weston Config:** `board/neuraparse/neuraos/rootfs_overlay/etc/xdg/weston/weston.ini`
- **Startup Script:** `board/neuraparse/neuraos/rootfs_overlay/usr/bin/start-weston.sh`
- **Init Script:** `board/neuraparse/neuraos/rootfs_overlay/etc/init.d/S99weston`
- **Dashboard Source:** `src/dashboard/main.cpp`

### Customization

#### Change Display Resolution

Edit `weston.ini`:
```ini
[output]
name=HDMI-A-1
mode=1920x1080@60  # Change this
```

#### Disable Auto-launch

Remove from `weston.ini`:
```ini
[autolaunch]
path=/usr/bin/neuraos-dashboard
```

#### Custom Theme

Edit `src/dashboard/main.cpp`:
```cpp
void applyDarkTheme() {
    setStyleSheet(
        "QMainWindow { background-color: #0D0D0D; }"  // Change colors
        // ...
    );
}
```

### Troubleshooting

**Weston fails to start:**
- Check `/var/log/weston.log`
- Try fbdev backend: `weston --backend=fbdev-backend.so`

**Dashboard doesn't appear:**
- Check if Qt5 was built: `ls /usr/lib/libQt5*`
- Run manually: `/usr/bin/neuraos-dashboard`

**No display in QEMU:**
- Make sure you used `run_qemu_gui.sh` (not headless)
- Check QEMU supports VirtIO GPU: `qemu-system-aarch64 -device help | grep virtio-gpu`

### Performance

**Minimal Configuration:**
- Kernel: ~8 MB
- Rootfs: ~150 MB (with GUI)
- RAM: 512 MB minimum, 1 GB recommended
- Boot time: < 5 seconds

**Full Configuration:**
- Rootfs: ~300 MB (with all AI frameworks)
- RAM: 1-2 GB recommended
- Boot time: < 10 seconds

### Next Steps

1. **Build the OS:** `./scripts/build_neuraos.sh`
2. **Test in QEMU:** `./scripts/run_qemu_gui.sh`
3. **Flash to SD card:** `dd if=output/images/sdcard.img of=/dev/sdX`
4. **Boot on Raspberry Pi 4**

Enjoy your AI-powered embedded OS with a beautiful GUI! 🚀

