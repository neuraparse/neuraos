################################################################################
#
# fastcdr - Fast CDR serialization library
#
################################################################################

# Fast CDR v2.3.5 (January 2026) - Latest stable
FASTCDR_VERSION = v2.3.5
FASTCDR_SITE = $(call github,eProsima,Fast-CDR,$(FASTCDR_VERSION))
FASTCDR_LICENSE = Apache-2.0
FASTCDR_LICENSE_FILES = LICENSE
FASTCDR_INSTALL_STAGING = YES
FASTCDR_INSTALL_TARGET = YES

FASTCDR_DEPENDENCIES = host-cmake

FASTCDR_CONF_OPTS = \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_BUILD_TYPE=Release

$(eval $(cmake-package))
