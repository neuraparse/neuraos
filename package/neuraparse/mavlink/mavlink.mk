################################################################################
#
# mavlink
#
################################################################################

MAVLINK_VERSION = 2.0.0
MAVLINK_SITE = $(call github,mavlink,mavlink,$(MAVLINK_VERSION))
MAVLINK_LICENSE = LGPL-3.0
MAVLINK_LICENSE_FILES = COPYING
MAVLINK_INSTALL_STAGING = YES
MAVLINK_INSTALL_TARGET = YES

MAVLINK_DEPENDENCIES = host-python3

# MAVLink is header-only, install headers
define MAVLINK_INSTALL_STAGING_CMDS
	$(INSTALL) -d $(STAGING_DIR)/usr/include/mavlink
	cp -r $(@D)/message_definitions $(STAGING_DIR)/usr/include/mavlink/
	if [ -d $(@D)/pymavlink/generator/C/include_v2.0 ]; then \
		cp -r $(@D)/pymavlink/generator/C/include_v2.0/* $(STAGING_DIR)/usr/include/mavlink/; \
	fi
endef

define MAVLINK_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/include/mavlink
	cp -r $(@D)/message_definitions $(TARGET_DIR)/usr/include/mavlink/
	if [ -d $(@D)/pymavlink/generator/C/include_v2.0 ]; then \
		cp -r $(@D)/pymavlink/generator/C/include_v2.0/* $(TARGET_DIR)/usr/include/mavlink/; \
	fi
endef

$(eval $(generic-package))

