################################################################################
#
# npi-init - NeuralOS init system
#
################################################################################

NPI_INIT_VERSION = 1.0.0
NPI_INIT_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/src/npi
NPI_INIT_SITE_METHOD = local
NPI_INIT_LICENSE = GPL-2.0-only
NPI_INIT_LICENSE_FILES = LICENSE
NPI_INIT_INSTALL_STAGING = NO
NPI_INIT_INSTALL_TARGET = YES

NPI_INIT_DEPENDENCIES = host-cmake

# Build configuration
NPI_INIT_CONF_OPTS += \
	-DCMAKE_BUILD_TYPE=Release

# Let CMake install under /usr to avoid conflicting with /sbin/init
NPI_INIT_CONF_OPTS += -DCMAKE_INSTALL_PREFIX=/usr

$(eval $(cmake-package))

