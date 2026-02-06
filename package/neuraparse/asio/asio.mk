################################################################################
#
# asio - C++ Network Library (Header-only)
#
################################################################################

# Asio 1.36.0 (August 2025) - Latest stable standalone version
ASIO_VERSION = asio-1-36-0
ASIO_SITE = $(call github,chriskohlhoff,asio,$(ASIO_VERSION))
ASIO_SOURCE = asio-$(ASIO_VERSION).tar.gz
ASIO_LICENSE = BSL-1.0
ASIO_LICENSE_FILES = asio/LICENSE_1_0.txt
ASIO_INSTALL_STAGING = YES
ASIO_INSTALL_TARGET = NO

# Asio is header-only, just install headers
define ASIO_INSTALL_STAGING_CMDS
	$(INSTALL) -d $(STAGING_DIR)/usr/include
	cp -r $(@D)/asio/include/asio $(STAGING_DIR)/usr/include/
	cp $(@D)/asio/include/asio.hpp $(STAGING_DIR)/usr/include/
endef

$(eval $(generic-package))
