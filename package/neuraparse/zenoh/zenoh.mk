################################################################################
#
# zenoh - Zero Overhead Pub/Sub, Store/Query and Compute Protocol
#
################################################################################

# Zenoh 1.1.0 (2026) - Next-gen pub/sub for ROS2 and robotics
ZENOH_VERSION = 1.1.0
ZENOH_SITE = $(call github,eclipse-zenoh,zenoh,$(ZENOH_VERSION))
ZENOH_LICENSE = Apache-2.0, EPL-2.0
ZENOH_LICENSE_FILES = LICENSE
ZENOH_INSTALL_STAGING = YES
ZENOH_INSTALL_TARGET = YES

# Zenoh is written in Rust
ZENOH_DEPENDENCIES = host-rustc

# Build options
ZENOH_CARGO_BUILD_OPTS = --release

# Install zenohd daemon and libraries
define ZENOH_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/target/$(RUSTC_TARGET_NAME)/release/zenohd \
		$(TARGET_DIR)/usr/bin/zenohd
	$(INSTALL) -D -m 0644 $(@D)/target/$(RUSTC_TARGET_NAME)/release/libzenoh*.so \
		$(TARGET_DIR)/usr/lib/
endef

define ZENOH_INSTALL_STAGING_CMDS
	$(INSTALL) -D -m 0644 $(@D)/target/$(RUSTC_TARGET_NAME)/release/libzenoh*.so \
		$(STAGING_DIR)/usr/lib/
endef

$(eval $(cargo-package))
