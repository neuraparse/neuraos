################################################################################
#
# ardupilot
#
################################################################################

# ArduPilot Copter 4.6.3 (January 2026) - Latest stable
ARDUPILOT_VERSION = Copter-4.6.3
ARDUPILOT_SITE = $(call github,ArduPilot,ardupilot,$(ARDUPILOT_VERSION))
ARDUPILOT_LICENSE = GPL-3.0
ARDUPILOT_LICENSE_FILES = COPYING.txt
ARDUPILOT_INSTALL_TARGET = YES

ARDUPILOT_DEPENDENCIES = mavlink

# Install pre-built binaries or headers only for now
define ARDUPILOT_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/share/ardupilot
	echo "ArduPilot $(ARDUPILOT_VERSION)" > $(TARGET_DIR)/usr/share/ardupilot/version
endef

$(eval $(generic-package))
