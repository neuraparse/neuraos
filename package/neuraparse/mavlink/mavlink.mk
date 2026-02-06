################################################################################
#
# mavlink
#
################################################################################

# MAVLink c_library_v2 (February 2026) - Auto-generated headers
MAVLINK_VERSION = master
MAVLINK_SITE = $(call github,mavlink,c_library_v2,$(MAVLINK_VERSION))
MAVLINK_LICENSE = LGPL-3.0
MAVLINK_LICENSE_FILES = COPYING
MAVLINK_INSTALL_STAGING = YES
MAVLINK_INSTALL_TARGET = YES

# MAVLink is header-only, install headers
define MAVLINK_INSTALL_STAGING_CMDS
	$(INSTALL) -d $(STAGING_DIR)/usr/include/mavlink
	cp -r $(@D)/* $(STAGING_DIR)/usr/include/mavlink/
endef

define MAVLINK_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/include/mavlink
	cp -r $(@D)/* $(TARGET_DIR)/usr/include/mavlink/
endef

$(eval $(generic-package))
