################################################################################
#
# xacc
#
################################################################################

# XACC 1.0.0 - Quantum programming framework
XACC_VERSION = 1.0.0
XACC_SITE = $(call github,eclipse,xacc,v$(XACC_VERSION))
XACC_LICENSE = EPL-2.0
XACC_LICENSE_FILES = LICENSE
XACC_INSTALL_STAGING = YES
XACC_INSTALL_TARGET = YES

XACC_DEPENDENCIES = host-cmake boost

XACC_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DXACC_BUILD_TESTS=OFF \
	-DXACC_BUILD_EXAMPLES=OFF

$(eval $(cmake-package))
