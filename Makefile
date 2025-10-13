#
# NeuralOS Main Makefile
# Version: 1.0.0-alpha
# Updated: October 2025
#

# Project configuration
PROJECT_NAME := NeuralOS
VERSION := 1.0.0-alpha
BUILD_DIR := build
OUTPUT_DIR := output
BUILDROOT_DIR := buildroot
CONFIGS_DIR := configs
SCRIPTS_DIR := scripts

# Default target
.DEFAULT_GOAL := help

# Buildroot configuration
BR2_EXTERNAL := $(CURDIR)
BUILDROOT_CONFIG := $(CONFIGS_DIR)/neuraos_defconfig
BUILDROOT_OUTPUT := $(OUTPUT_DIR)/buildroot

# CMake configuration
CMAKE_BUILD_DIR := $(BUILD_DIR)/cmake
CMAKE_INSTALL_PREFIX := $(OUTPUT_DIR)/install

# Detect number of CPU cores
NPROC := $(shell nproc 2>/dev/null || echo 4)
MAKEFLAGS += -j$(NPROC)

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_CYAN := \033[36m

# Helper function for colored output
define print_header
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)==> $(1)$(COLOR_RESET)"
endef

define print_success
	@echo "$(COLOR_BOLD)$(COLOR_GREEN)✓ $(1)$(COLOR_RESET)"
endef

define print_warning
	@echo "$(COLOR_BOLD)$(COLOR_YELLOW)⚠ $(1)$(COLOR_RESET)"
endef

#
# Help target
#
.PHONY: help
help:
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)"
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║                    NeuralOS Build System                      ║"
	@echo "║                    Version: $(VERSION)                        ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo "$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)Main Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)all$(COLOR_RESET)                  - Build complete NeuralOS system"
	@echo "  $(COLOR_GREEN)buildroot$(COLOR_RESET)            - Build Buildroot-based system"
	@echo "  $(COLOR_GREEN)npie$(COLOR_RESET)                 - Build NeuraParse Inference Engine"
	@echo "  $(COLOR_GREEN)tools$(COLOR_RESET)                - Build development tools"
	@echo "  $(COLOR_GREEN)examples$(COLOR_RESET)             - Build example applications"
	@echo "  $(COLOR_GREEN)sdk$(COLOR_RESET)                  - Generate SDK for cross-compilation"
	@echo ""
	@echo "$(COLOR_BOLD)Configuration Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_CYAN)menuconfig$(COLOR_RESET)           - Configure Buildroot"
	@echo "  $(COLOR_CYAN)linux-menuconfig$(COLOR_RESET)     - Configure Linux kernel"
	@echo "  $(COLOR_CYAN)busybox-menuconfig$(COLOR_RESET)   - Configure BusyBox"
	@echo "  $(COLOR_CYAN)savedefconfig$(COLOR_RESET)        - Save current configuration"
	@echo ""
	@echo "$(COLOR_BOLD)Board-Specific Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_YELLOW)rpi4$(COLOR_RESET)                 - Configure for Raspberry Pi 4"
	@echo "  $(COLOR_YELLOW)jetson-nano$(COLOR_RESET)          - Configure for NVIDIA Jetson Nano"
	@echo "  $(COLOR_YELLOW)x86_64$(COLOR_RESET)               - Configure for x86_64 PC"
	@echo "  $(COLOR_YELLOW)riscv$(COLOR_RESET)                - Configure for RISC-V"
	@echo ""
	@echo "$(COLOR_BOLD)Utility Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_BLUE)clean$(COLOR_RESET)                - Clean build artifacts"
	@echo "  $(COLOR_BLUE)distclean$(COLOR_RESET)            - Clean everything including downloads"
	@echo "  $(COLOR_BLUE)flash$(COLOR_RESET)                - Flash image to SD card/device"
	@echo "  $(COLOR_BLUE)test$(COLOR_RESET)                 - Run test suite"
	@echo "  $(COLOR_BLUE)docs$(COLOR_RESET)                 - Generate documentation"
	@echo "  $(COLOR_BLUE)help$(COLOR_RESET)                 - Show this help message"
	@echo ""

#
# Main build targets
#
.PHONY: all
all: buildroot npie tools
	$(call print_success,"NeuralOS build complete!")
	@echo ""
	@echo "Output files:"
	@echo "  - Root filesystem: $(BUILDROOT_OUTPUT)/images/rootfs.squashfs"
	@echo "  - Kernel image: $(BUILDROOT_OUTPUT)/images/Image"
	@echo "  - SD card image: $(BUILDROOT_OUTPUT)/images/sdcard.img"
	@echo ""

.PHONY: buildroot
buildroot: $(BUILDROOT_DIR)/.config
	$(call print_header,"Building Buildroot system")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL)
	$(call print_success,"Buildroot build complete")

.PHONY: npie
npie: cmake-configure
	$(call print_header,"Building NeuraParse Inference Engine")
	@cmake --build $(CMAKE_BUILD_DIR) --target npie
	$(call print_success,"NPIE build complete")

.PHONY: tools
tools: cmake-configure
	$(call print_header,"Building development tools")
	@cmake --build $(CMAKE_BUILD_DIR) --target tools
	$(call print_success,"Tools build complete")

.PHONY: examples
examples: cmake-configure
	$(call print_header,"Building example applications")
	@cmake --build $(CMAKE_BUILD_DIR) --target examples
	$(call print_success,"Examples build complete")

.PHONY: sdk
sdk: buildroot
	$(call print_header,"Generating SDK")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) sdk
	$(call print_success,"SDK generated at $(BUILDROOT_OUTPUT)/host")

#
# Configuration targets
#
$(BUILDROOT_DIR)/.config:
	$(call print_header,"Initializing Buildroot configuration")
	@mkdir -p $(BUILDROOT_OUTPUT)
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) $(BUILDROOT_CONFIG)

.PHONY: menuconfig
menuconfig: $(BUILDROOT_DIR)/.config
	$(call print_header,"Configuring Buildroot")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) menuconfig

.PHONY: linux-menuconfig
linux-menuconfig: $(BUILDROOT_DIR)/.config
	$(call print_header,"Configuring Linux kernel")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) linux-menuconfig

.PHONY: busybox-menuconfig
busybox-menuconfig: $(BUILDROOT_DIR)/.config
	$(call print_header,"Configuring BusyBox")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) busybox-menuconfig

.PHONY: savedefconfig
savedefconfig:
	$(call print_header,"Saving configuration")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) savedefconfig
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) linux-update-defconfig
	$(call print_success,"Configuration saved")

#
# Board-specific configurations
#
.PHONY: rpi4
rpi4:
	$(call print_header,"Configuring for Raspberry Pi 4")
	@mkdir -p $(BUILDROOT_OUTPUT)
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) neuraos_rpi4_defconfig
	$(call print_success,"Raspberry Pi 4 configuration loaded")

.PHONY: jetson-nano
jetson-nano:
	$(call print_header,"Configuring for NVIDIA Jetson Nano")
	@mkdir -p $(BUILDROOT_OUTPUT)
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) neuraos_jetson_defconfig
	$(call print_success,"Jetson Nano configuration loaded")

.PHONY: x86_64
x86_64:
	$(call print_header,"Configuring for x86_64 PC")
	@mkdir -p $(BUILDROOT_OUTPUT)
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) neuraos_x86_64_defconfig
	$(call print_success,"x86_64 configuration loaded")

.PHONY: riscv
riscv:
	$(call print_header,"Configuring for RISC-V")
	@mkdir -p $(BUILDROOT_OUTPUT)
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) BR2_EXTERNAL=$(BR2_EXTERNAL) neuraos_riscv_defconfig
	$(call print_success,"RISC-V configuration loaded")

#
# CMake targets
#
.PHONY: cmake-configure
cmake-configure:
	@mkdir -p $(CMAKE_BUILD_DIR)
	@if [ ! -f $(CMAKE_BUILD_DIR)/Makefile ]; then \
		$(call print_header,"Configuring CMake"); \
		cd $(CMAKE_BUILD_DIR) && cmake $(CURDIR) \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX) \
			-DBUILD_NPIE=ON \
			-DBUILD_TOOLS=ON \
			-DBUILD_EXAMPLES=ON \
			-DBUILD_TESTS=ON; \
		$(call print_success,"CMake configuration complete"); \
	fi

.PHONY: cmake-clean
cmake-clean:
	$(call print_header,"Cleaning CMake build")
	@rm -rf $(CMAKE_BUILD_DIR)
	$(call print_success,"CMake build cleaned")

#
# Testing
#
.PHONY: test
test: cmake-configure
	$(call print_header,"Running test suite")
	@cd $(CMAKE_BUILD_DIR) && ctest --output-on-failure
	$(call print_success,"Tests complete")

#
# Documentation
#
.PHONY: docs
docs:
	$(call print_header,"Generating documentation")
	@if command -v doxygen >/dev/null 2>&1; then \
		doxygen Doxyfile; \
		$(call print_success,"Documentation generated in docs/html"); \
	else \
		$(call print_warning,"Doxygen not found, skipping documentation"); \
	fi

#
# Flashing
#
.PHONY: flash
flash:
	@if [ -z "$(DEVICE)" ]; then \
		echo "$(COLOR_BOLD)$(COLOR_YELLOW)Usage: make flash DEVICE=/dev/sdX$(COLOR_RESET)"; \
		echo "Available devices:"; \
		lsblk -d -o NAME,SIZE,TYPE,MOUNTPOINT | grep -E "disk|NAME"; \
		exit 1; \
	fi
	$(call print_header,"Flashing to $(DEVICE)")
	@echo "$(COLOR_BOLD)$(COLOR_YELLOW)WARNING: This will erase all data on $(DEVICE)!$(COLOR_RESET)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		sudo dd if=$(BUILDROOT_OUTPUT)/images/sdcard.img of=$(DEVICE) bs=4M status=progress conv=fsync; \
		sync; \
		$(call print_success,"Flashing complete"); \
	else \
		echo "Aborted."; \
	fi

#
# Cleaning
#
.PHONY: clean
clean: cmake-clean
	$(call print_header,"Cleaning build artifacts")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) clean 2>/dev/null || true
	@rm -rf $(BUILD_DIR)
	$(call print_success,"Clean complete")

.PHONY: distclean
distclean:
	$(call print_header,"Deep cleaning (including downloads)")
	@$(MAKE) -C $(BUILDROOT_DIR) O=$(BUILDROOT_OUTPUT) distclean 2>/dev/null || true
	@rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
	$(call print_success,"Distclean complete")

#
# Development helpers
#
.PHONY: setup
setup:
	$(call print_header,"Setting up development environment")
	@bash $(SCRIPTS_DIR)/setup_environment.sh
	$(call print_success,"Environment setup complete")

.PHONY: check-deps
check-deps:
	$(call print_header,"Checking dependencies")
	@bash $(SCRIPTS_DIR)/check_dependencies.sh

.PHONY: version
version:
	@echo "$(PROJECT_NAME) version $(VERSION)"

.PHONY: info
info:
	@echo "$(COLOR_BOLD)Project Information:$(COLOR_RESET)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Build directory: $(BUILD_DIR)"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@echo "  Buildroot directory: $(BUILDROOT_DIR)"
	@echo "  CPU cores: $(NPROC)"
	@echo ""
	@echo "$(COLOR_BOLD)Build Configuration:$(COLOR_RESET)"
	@if [ -f $(BUILDROOT_OUTPUT)/.config ]; then \
		echo "  Buildroot configured: Yes"; \
		grep "^BR2_ARCH=" $(BUILDROOT_OUTPUT)/.config || true; \
		grep "^BR2_LINUX_KERNEL_VERSION=" $(BUILDROOT_OUTPUT)/.config || true; \
	else \
		echo "  Buildroot configured: No"; \
	fi

# Prevent make from deleting intermediate files
.SECONDARY:

# Declare phony targets
.PHONY: all buildroot npie tools examples sdk menuconfig linux-menuconfig \
        busybox-menuconfig savedefconfig rpi4 jetson-nano x86_64 riscv \
        cmake-configure cmake-clean test docs flash clean distclean setup \
        check-deps version info help

