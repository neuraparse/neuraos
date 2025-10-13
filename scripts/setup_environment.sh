#!/bin/bash
#
# NeuralOS Development Environment Setup Script
# Version: 1.0.0-alpha
# Updated: October 2025
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BOLD}${CYAN}==>${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        print_error "Cannot detect OS"
        exit 1
    fi
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. This is not recommended for development."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install dependencies for Ubuntu/Debian
install_deps_debian() {
    print_header "Installing dependencies for Debian/Ubuntu"
    
    sudo apt-get update
    
    # Build essentials
    sudo apt-get install -y \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        ninja-build \
        ccache \
        git \
        wget \
        curl \
        rsync \
        bc \
        bison \
        flex \
        libssl-dev \
        libncurses5-dev \
        libelf-dev \
        device-tree-compiler \
        u-boot-tools \
        qemu-system-arm \
        qemu-system-x86 \
        qemu-user-static \
        debootstrap
    
    # Python and tools
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-numpy \
        python3-pil
    
    # Cross-compilation tools
    sudo apt-get install -y \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        gcc-riscv64-linux-gnu \
        g++-riscv64-linux-gnu
    
    # Additional tools
    sudo apt-get install -y \
        cpio \
        unzip \
        file \
        patch \
        perl \
        tar \
        gzip \
        bzip2 \
        xz-utils \
        zstd \
        lz4 \
        dosfstools \
        mtools \
        genext2fs \
        libarchive-tools \
        squashfs-tools
    
    # Documentation tools
    sudo apt-get install -y \
        doxygen \
        graphviz \
        texinfo
    
    print_success "Dependencies installed successfully"
}

# Install dependencies for Fedora/RHEL
install_deps_fedora() {
    print_header "Installing dependencies for Fedora/RHEL"
    
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y \
        gcc \
        gcc-c++ \
        make \
        cmake \
        ninja-build \
        ccache \
        git \
        wget \
        curl \
        rsync \
        bc \
        bison \
        flex \
        openssl-devel \
        ncurses-devel \
        elfutils-libelf-devel \
        dtc \
        uboot-tools \
        qemu-system-arm \
        qemu-system-x86 \
        python3 \
        python3-pip \
        python3-devel \
        python3-numpy \
        cpio \
        unzip \
        file \
        patch \
        perl \
        tar \
        gzip \
        bzip2 \
        xz \
        zstd \
        lz4 \
        dosfstools \
        mtools \
        squashfs-tools \
        doxygen \
        graphviz \
        texinfo
    
    print_success "Dependencies installed successfully"
}

# Install dependencies for Arch Linux
install_deps_arch() {
    print_header "Installing dependencies for Arch Linux"
    
    sudo pacman -Syu --needed --noconfirm \
        base-devel \
        gcc \
        cmake \
        ninja \
        ccache \
        git \
        wget \
        curl \
        rsync \
        bc \
        bison \
        flex \
        openssl \
        ncurses \
        libelf \
        dtc \
        uboot-tools \
        qemu-system-arm \
        qemu-system-x86 \
        python \
        python-pip \
        python-numpy \
        python-pillow \
        cpio \
        unzip \
        file \
        patch \
        perl \
        tar \
        gzip \
        bzip2 \
        xz \
        zstd \
        lz4 \
        dosfstools \
        mtools \
        squashfs-tools \
        doxygen \
        graphviz \
        texinfo
    
    print_success "Dependencies installed successfully"
}

# Clone Buildroot
setup_buildroot() {
    print_header "Setting up Buildroot"
    
    if [ -d "buildroot" ]; then
        print_warning "Buildroot directory already exists"
        read -p "Re-clone? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf buildroot
        else
            return
        fi
    fi
    
    git clone --depth 1 --branch 2025.08 \
        https://gitlab.com/buildroot.org/buildroot.git buildroot
    
    print_success "Buildroot cloned successfully"
}

# Setup Python virtual environment
setup_python_venv() {
    print_header "Setting up Python virtual environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        return
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    pip install --upgrade pip setuptools wheel
    pip install numpy pillow pyyaml
    
    # Install TensorFlow Lite tools
    pip install tflite-runtime
    
    # Install ONNX tools
    pip install onnx onnxruntime
    
    deactivate
    
    print_success "Python virtual environment created"
}

# Create build directories
create_directories() {
    print_header "Creating build directories"
    
    mkdir -p build/cmake
    mkdir -p output/buildroot
    mkdir -p output/images
    mkdir -p output/install
    mkdir -p downloads
    
    print_success "Directories created"
}

# Setup git hooks
setup_git_hooks() {
    print_header "Setting up git hooks"
    
    if [ -d ".git" ]; then
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for NeuralOS

# Check for trailing whitespace
git diff --check --cached
if [ $? -ne 0 ]; then
    echo "Error: Trailing whitespace detected"
    exit 1
fi

# Check C/C++ formatting (if clang-format is available)
if command -v clang-format >/dev/null 2>&1; then
    for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(c|cpp|h|hpp)$'); do
        clang-format -i "$file"
        git add "$file"
    done
fi

exit 0
EOF
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks installed"
    else
        print_warning "Not a git repository, skipping git hooks"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║         NeuralOS Development Environment Ready!               ║${NC}"
    echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo "  1. Configure for your target board:"
    echo "     ${CYAN}make rpi4${NC}              # For Raspberry Pi 4"
    echo "     ${CYAN}make jetson-nano${NC}       # For NVIDIA Jetson Nano"
    echo "     ${CYAN}make x86_64${NC}            # For x86_64 PC"
    echo ""
    echo "  2. Build the system:"
    echo "     ${CYAN}make all${NC}               # Build everything"
    echo ""
    echo "  3. Flash to device:"
    echo "     ${CYAN}make flash DEVICE=/dev/sdX${NC}"
    echo ""
    echo -e "${BOLD}Documentation:${NC}"
    echo "  - Getting Started: docs/getting_started.md"
    echo "  - API Reference: docs/api_reference.md"
    echo "  - Hardware Support: docs/hardware_support.md"
    echo ""
    echo -e "${BOLD}Activate Python environment:${NC}"
    echo "  ${CYAN}source venv/bin/activate${NC}"
    echo ""
}

# Main function
main() {
    echo -e "${BOLD}${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          NeuralOS Development Environment Setup               ║"
    echo "║                  Version: 1.0.0-alpha                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_root
    detect_os
    
    print_header "Detected OS: $OS $OS_VERSION"
    
    # Install dependencies based on OS
    case $OS in
        ubuntu|debian)
            install_deps_debian
            ;;
        fedora|rhel|centos)
            install_deps_fedora
            ;;
        arch|manjaro)
            install_deps_arch
            ;;
        *)
            print_error "Unsupported OS: $OS"
            print_warning "Please install dependencies manually"
            ;;
    esac
    
    setup_buildroot
    setup_python_venv
    create_directories
    setup_git_hooks
    
    print_summary
}

# Run main function
main "$@"

