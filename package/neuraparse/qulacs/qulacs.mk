################################################################################
#
# Qulacs - Fast Quantum Circuit Simulator
#
################################################################################

# Qulacs v0.6.13 (February 2025)
QULACS_VERSION = 0.6.13
QULACS_SITE = $(call github,qulacs,qulacs,v$(QULACS_VERSION))
QULACS_LICENSE = MIT
QULACS_LICENSE_FILES = LICENSE
QULACS_INSTALL_STAGING = YES
QULACS_INSTALL_TARGET = YES

QULACS_DEPENDENCIES = host-cmake boost

QULACS_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTS=OFF \
	-DUSE_GPU=No

# Enable OpenMP for multi-threading
ifeq ($(BR2_TOOLCHAIN_HAS_OPENMP),y)
QULACS_CONF_OPTS += -DUSE_OMP=Yes
else
QULACS_CONF_OPTS += -DUSE_OMP=No
endif

$(eval $(cmake-package))
