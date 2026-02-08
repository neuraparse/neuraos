# NeuralOS GUI Build Guide

## Qt5 QML Desktop Shell

NeuralOS includes a full-featured Qt5 QML desktop environment with glassmorphism design, 33 built-in applications, 14 C++ backend managers, and integration with NPIE v2.0.0 (12 fully implemented AI/ML/Quantum backends).

### Features

- **Qt5 QML Desktop Shell** - Glassmorphism UI with dark/light theme
- **Wayland/Weston Compositor** - Modern display server
- **33 Built-in Applications** - AI, System, Quantum, Utilities, Media, Defense
- **14 C++ Backend Managers** - SystemInfo, NPIE, NPU, AI Bus, AI Memory, Quantum, etc.
- **Hardware Acceleration** - Mesa3D, DRM/KMS, VirtIO GPU, Vulkan
- **Command Palette** - Natural language OS control (Ctrl+K)

### Desktop Shell Components

1. **Floating Dock Taskbar** - App indicators with running state
2. **Start Menu** - Search, pinned apps, categorized app grid
3. **Notification Center** - Quick toggles, brightness/volume sliders
4. **Command Palette** - Ctrl+K overlay for natural language commands
5. **Window Manager** - Drag, resize, minimize, maximize, close
6. **Desktop Widgets** - Clock, system stats, weather, media, calendar

### Applications (33 total)

| Category | Count | Apps |
|----------|-------|------|
| System | 8 | System Monitor, Terminal, File Manager, Settings, Task Manager, Package Manager, Network Center, Ecosystem |
| AI & ML | 9 | Neural Studio, AI Agent Hub, AI Assistant, NPU Control, AI Bus, AI Memory, Automation Studio, MCP Hub, Knowledge Base |
| Utilities | 7 | Calculator, Text Editor, Notes, Calendar, Clock, Weather, Photos |
| Media | 3 | Music Player, Video Player, Image Viewer |
| Internet | 2 | Web Browser, App Store |
| Defense | 2 | Drone Command Center, Defense Monitor |
| Quantum | 1 | Quantum Lab (statevector simulation, 13 gates, 5 backends) |

### Build Instructions

#### 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt install qtbase5-dev qtdeclarative5-dev qtquickcontrols2-5-dev \
    libqt5svg5-dev qml-module-qtquick2 qml-module-qtquick-controls2 \
    qml-module-qtquick-layouts cmake g++ make

# Fedora
sudo dnf install qt5-qtbase-devel qt5-qtdeclarative-devel \
    qt5-qtquickcontrols2-devel qt5-qtsvg-devel cmake gcc-c++ make
```

#### 2. Build with CMake (Recommended)

```bash
cd neuraos
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The dashboard binary will be at `build/src/dashboard/neuraos-dashboard`.

#### 3. Run

```bash
# Run directly (requires display server)
./build/src/dashboard/neuraos-dashboard

# Or with Wayland
export QT_QPA_PLATFORM=wayland
./build/src/dashboard/neuraos-dashboard

# Or with X11/XCB
export QT_QPA_PLATFORM=xcb
./build/src/dashboard/neuraos-dashboard
```

#### 4. Full Buildroot Build (Embedded Image)

```bash
# Build complete OS with GUI support
./scripts/build_neuraos.sh

# Test in QEMU with GUI
./scripts/run_qemu_gui.sh

# Or headless
./scripts/run_qemu_headless.sh
```

### Backend Managers

The desktop shell uses 14 C++ backend managers exposed to QML:

| Manager | QML Context | Description |
|---------|-------------|-------------|
| SystemInfo | `SystemInfo` | CPU, memory, disk, OS info |
| NPIEBridge | `NPIE` | Inference engine bridge |
| NPUMonitor | `NPUMonitor` | NPU hardware monitoring |
| ProcessManager | `ProcessManager` | Process lifecycle |
| NetworkManager | `NetworkManager` | Network configuration |
| SettingsManager | `Settings` | App settings storage |
| AIBusManager | `AIBus` | Multi-model orchestration |
| AIMemoryManager | `AIMemory` | Cross-app AI memory |
| CommandPalette | `CommandEngine` | Natural language commands |
| AutomationManager | `Automation` | Workflow recording/replay |
| MCPManager | `MCP` | Model Context Protocol |
| KnowledgeManager | `Knowledge` | RAG document search |
| EcosystemManager | `Ecosystem` | Multi-device management |
| QuantumManager | `Quantum` | Quantum circuit simulation |

### Boot Sequence (Embedded)

1. **Kernel Boot** - Linux 6.12 LTS with PREEMPT_RT
2. **Init System (NPI)** - Fast boot, minimal overhead
3. **Weston Compositor** - Wayland display server
4. **neuraos-dashboard** - Qt5 QML application launches automatically

### Configuration

**Display Resolution:**
Edit `weston.ini`:
```ini
[output]
name=HDMI-A-1
mode=1920x1080@60
```

**Auto-launch:**
The dashboard auto-launches via `weston.ini`:
```ini
[autolaunch]
path=/usr/bin/neuraos-dashboard
```

### Performance

| Configuration | Rootfs Size | RAM | Boot Time |
|--------------|-------------|-----|-----------|
| Minimal (no AI libs) | ~150 MB | 512 MB min | < 5 sec |
| Full (all AI frameworks) | ~300 MB | 1-2 GB | < 10 sec |

### Troubleshooting

**Qt5 not found during build:**
```bash
# Ensure Qt5 dev packages are installed
apt list --installed | grep qt5
# Install if missing
sudo apt install qtbase5-dev qtdeclarative5-dev qtquickcontrols2-5-dev
```

**Dashboard doesn't start:**
```bash
# Check Qt platform plugin
export QT_DEBUG_PLUGINS=1
./neuraos-dashboard
# Try different platform
QT_QPA_PLATFORM=offscreen ./neuraos-dashboard
```

**No display in QEMU:**
```bash
# Use GUI script with VirtIO GPU
./scripts/run_qemu_gui.sh
# Verify VirtIO GPU support
qemu-system-x86_64 -device help | grep virtio-gpu
```
