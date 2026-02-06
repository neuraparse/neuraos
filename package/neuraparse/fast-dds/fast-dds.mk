################################################################################
#
# fast-dds (eProsima Fast DDS)
#
################################################################################

# Fast-DDS v3.4.2 (January 2025) - includes security fixes
FAST_DDS_VERSION = 3.4.2
FAST_DDS_SITE = $(call github,eProsima,Fast-DDS,v$(FAST_DDS_VERSION))
FAST_DDS_LICENSE = Apache-2.0
FAST_DDS_LICENSE_FILES = LICENSE
FAST_DDS_INSTALL_STAGING = YES
FAST_DDS_INSTALL_TARGET = YES

# Complete dependency list for Fast-DDS
FAST_DDS_DEPENDENCIES = host-cmake openssl fastcdr foonathan-memory asio tinyxml2

FAST_DDS_CONF_OPTS = \
	-DBUILD_SHARED_LIBS=ON \
	-DCOMPILE_EXAMPLES=OFF \
	-DCOMPILE_TOOLS=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DTHIRDPARTY=OFF

# Enable DDS Security
ifeq ($(BR2_PACKAGE_FAST_DDS_SECURITY),y)
FAST_DDS_CONF_OPTS += -DSECURITY=ON
else
FAST_DDS_CONF_OPTS += -DSECURITY=OFF
endif

# Enable Shared Memory Transport
ifeq ($(BR2_PACKAGE_FAST_DDS_SHM),y)
FAST_DDS_CONF_OPTS += -DSHM_TRANSPORT_DEFAULT=ON
else
FAST_DDS_CONF_OPTS += -DSHM_TRANSPORT_DEFAULT=OFF
endif

$(eval $(cmake-package))

