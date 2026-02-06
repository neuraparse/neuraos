################################################################################
#
# mavsdk
#
################################################################################

# MAVSDK v3.14.0 (January 2026) - Latest stable
MAVSDK_VERSION = v3.14.0
MAVSDK_SITE = $(call github,mavlink,MAVSDK,$(MAVSDK_VERSION))
MAVSDK_LICENSE = BSD-3-Clause
MAVSDK_LICENSE_FILES = LICENSE
MAVSDK_INSTALL_STAGING = YES
MAVSDK_INSTALL_TARGET = YES

MAVSDK_DEPENDENCIES = host-cmake mavlink xz jsoncpp tinyxml2

MAVSDK_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTS=OFF \
	-DBUILD_SHARED_LIBS=ON \
	-DSUPERBUILD=OFF \
	-DBUILD_WITHOUT_CURL=ON \
	-DCMAKE_DISABLE_FIND_PACKAGE_LibLZMA=ON

$(eval $(cmake-package))
