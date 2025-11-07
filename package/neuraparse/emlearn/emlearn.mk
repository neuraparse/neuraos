################################################################################
#
# emlearn - Efficient Machine Learning for Embedded Systems
#
################################################################################

EMLEARN_VERSION = 0.21.1
EMLEARN_SITE = $(call github,emlearn,emlearn,$(EMLEARN_VERSION))
EMLEARN_LICENSE = MIT
EMLEARN_LICENSE_FILES = LICENSE
EMLEARN_INSTALL_STAGING = YES
EMLEARN_INSTALL_TARGET = YES

# emlearn is header-only, but we build examples and tools
EMLEARN_DEPENDENCIES = host-python3

# Install headers
define EMLEARN_INSTALL_STAGING_CMDS
	$(INSTALL) -d $(STAGING_DIR)/usr/include/emlearn
	cp -r $(@D)/emlearn/*.h $(STAGING_DIR)/usr/include/emlearn/
endef

# Install to target
define EMLEARN_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/include/emlearn
	cp -r $(@D)/emlearn/*.h $(TARGET_DIR)/usr/include/emlearn/
endef

$(eval $(generic-package))

