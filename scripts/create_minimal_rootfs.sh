#!/bin/bash
#
# Create a minimal bootable rootfs for NeuralOS testing
# This creates a quick test image without full Buildroot build
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ROOTFS_DIR="$PROJECT_ROOT/output/rootfs"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Creating minimal NeuralOS rootfs..."

# Clean and create rootfs directory
rm -rf "$ROOTFS_DIR"
mkdir -p "$ROOTFS_DIR"

# Create basic directory structure
echo "Creating directory structure..."
mkdir -p "$ROOTFS_DIR"/{bin,sbin,etc,proc,sys,dev,tmp,var,usr/{bin,sbin,lib},lib,root,home}

# Copy our binaries
echo "Copying NeuralOS binaries..."
if [ -d "$BUILD_DIR" ]; then
    # Copy NPI init
    if [ -f "$BUILD_DIR/src/npi/npi" ]; then
        cp "$BUILD_DIR/src/npi/npi" "$ROOTFS_DIR/sbin/init"
        chmod +x "$ROOTFS_DIR/sbin/init"
    fi
    
    # Copy NPIE library
    if [ -f "$BUILD_DIR/src/npie/libnpie.a" ]; then
        cp "$BUILD_DIR/src/npie/libnpie.a" "$ROOTFS_DIR/usr/lib/"
    fi
    
    # Copy tools
    if [ -d "$BUILD_DIR/tools" ]; then
        find "$BUILD_DIR/tools" -type f -executable -exec cp {} "$ROOTFS_DIR/usr/bin/" \;
    fi
    
    # Copy examples
    if [ -d "$BUILD_DIR/examples" ]; then
        mkdir -p "$ROOTFS_DIR/usr/share/neuraos/examples"
        find "$BUILD_DIR/examples" -type f -executable -exec cp {} "$ROOTFS_DIR/usr/share/neuraos/examples/" \;
    fi
else
    echo "Warning: Build directory not found. Run 'cmake --build build' first."
fi

# Create basic config files
echo "Creating configuration files..."

# /etc/inittab (for BusyBox init fallback)
cat > "$ROOTFS_DIR/etc/inittab" << 'EOF'
::sysinit:/etc/init.d/rcS
::respawn:/sbin/getty -L console 0 vt100
::ctrlaltdel:/sbin/reboot
::shutdown:/bin/umount -a -r
EOF

# /etc/fstab
cat > "$ROOTFS_DIR/etc/fstab" << 'EOF'
# <file system> <mount point>   <type>  <options>       <dump>  <pass>
proc            /proc           proc    defaults        0       0
sysfs           /sys            sysfs   defaults        0       0
devtmpfs        /dev            devtmpfs mode=0755      0       0
tmpfs           /tmp            tmpfs   defaults        0       0
EOF

# /etc/hostname
echo "neuraos" > "$ROOTFS_DIR/etc/hostname"

# /etc/issue
cat > "$ROOTFS_DIR/etc/issue" << 'EOF'

 _   _                      _  ___  ____  
| \ | | ___ _   _ _ __ __ _| |/ _ \/ ___| 
|  \| |/ _ \ | | | '__/ _` | | | | \___ \ 
| |\  |  __/ |_| | | | (_| | | |_| |___) |
|_| \_|\___|\__,_|_|  \__,_|_|\___/|____/ 
                                          
NeuralOS v1.0.0-alpha - AI-Native Embedded OS

EOF

# /etc/passwd
cat > "$ROOTFS_DIR/etc/passwd" << 'EOF'
root:x:0:0:root:/root:/bin/sh
EOF

# /etc/group
cat > "$ROOTFS_DIR/etc/group" << 'EOF'
root:x:0:
EOF

# Create init script directory
mkdir -p "$ROOTFS_DIR/etc/init.d"

# /etc/init.d/rcS
cat > "$ROOTFS_DIR/etc/init.d/rcS" << 'EOF'
#!/bin/sh

echo "Starting NeuralOS..."

# Mount filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev

# Create device nodes if needed
[ -c /dev/null ] || mknod -m 666 /dev/null c 1 3
[ -c /dev/zero ] || mknod -m 666 /dev/zero c 1 5
[ -c /dev/console ] || mknod -m 600 /dev/console c 5 1

# Set hostname
hostname -F /etc/hostname

# Show banner
cat /etc/issue

echo "NeuralOS boot complete!"
echo ""
EOF

chmod +x "$ROOTFS_DIR/etc/init.d/rcS"

# Create a simple init if NPI is not available
if [ ! -f "$ROOTFS_DIR/sbin/init" ]; then
    echo "Creating fallback init script..."
    cat > "$ROOTFS_DIR/sbin/init" << 'EOF'
#!/bin/sh
# Fallback init for NeuralOS

exec /etc/init.d/rcS
exec /bin/sh
EOF
    chmod +x "$ROOTFS_DIR/sbin/init"
fi

# Set permissions
echo "Setting permissions..."
chmod 755 "$ROOTFS_DIR"
chmod 1777 "$ROOTFS_DIR/tmp"

echo ""
echo "Minimal rootfs created at: $ROOTFS_DIR"
echo ""
echo "To create initramfs:"
echo "  cd $ROOTFS_DIR"
echo "  find . | cpio -H newc -o | gzip > ../initramfs.cpio.gz"
echo ""
echo "Directory structure:"
ls -la "$ROOTFS_DIR"

