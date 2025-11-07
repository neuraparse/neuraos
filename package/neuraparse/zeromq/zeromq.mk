################################################################################
#
# zeromq (libzmq)
#
################################################################################

ZEROMQ_VERSION = 4.3.6
ZEROMQ_SITE = $(call github,zeromq,libzmq,v$(ZEROMQ_VERSION))
ZEROMQ_LICENSE = MPL-2.0
ZEROMQ_LICENSE_FILES = LICENSE
ZEROMQ_INSTALL_STAGING = YES
ZEROMQ_INSTALL_TARGET = YES

ZEROMQ_DEPENDENCIES = host-pkgconf

ZEROMQ_CONF_OPTS = \
	-DBUILD_SHARED=ON \
	-DBUILD_STATIC=OFF \
	-DBUILD_TESTS=OFF \
	-DENABLE_DRAFTS=OFF \
	-DCMAKE_BUILD_TYPE=Release

# Enable CURVE security
ifeq ($(BR2_PACKAGE_ZEROMQ_CURVE),y)
ZEROMQ_CONF_OPTS += -DWITH_LIBSODIUM=ON
ZEROMQ_DEPENDENCIES += libsodium
else
ZEROMQ_CONF_OPTS += -DWITH_LIBSODIUM=OFF
endif

$(eval $(cmake-package))

