################################################################################
#
# QuEST - Quantum Exact Simulation Toolkit
#
################################################################################

# QuEST v4.2.0 (October 2024) - Best for embedded systems
# Zero external dependencies, pure C/C++ implementation
QUEST_VERSION = 4.2.0
QUEST_SITE = $(call github,QuEST-Kit,QuEST,v$(QUEST_VERSION))
QUEST_LICENSE = MIT
QUEST_LICENSE_FILES = LICENSE
QUEST_INSTALL_STAGING = YES
QUEST_INSTALL_TARGET = YES

QUEST_DEPENDENCIES = host-cmake

QUEST_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DTESTING=OFF \
	-DBUILD_EXAMPLES=OFF

# Enable OpenMP for multi-threading if available
ifeq ($(BR2_TOOLCHAIN_HAS_OPENMP),y)
QUEST_CONF_OPTS += -DENABLE_MULTITHREADING=ON
else
QUEST_CONF_OPTS += -DENABLE_MULTITHREADING=OFF
endif

# Enable MPI for distributed computing if available
ifeq ($(BR2_PACKAGE_OPENMPI),y)
QUEST_CONF_OPTS += -DDISTRIBUTED=ON
QUEST_DEPENDENCIES += openmpi
else
QUEST_CONF_OPTS += -DDISTRIBUTED=OFF
endif

# Precision configuration
QUEST_CONF_OPTS += -DPRECISION=2

$(eval $(cmake-package))
