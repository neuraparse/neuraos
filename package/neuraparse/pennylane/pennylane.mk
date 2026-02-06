################################################################################
#
# pennylane
#
################################################################################

# PennyLane Lightning v0.44.0 (January 2026) - Quantum ML
PENNYLANE_VERSION = v0.44.0
PENNYLANE_SITE = $(call github,PennyLaneAI,pennylane-lightning,$(PENNYLANE_VERSION))
PENNYLANE_LICENSE = Apache-2.0
PENNYLANE_LICENSE_FILES = LICENSE
PENNYLANE_INSTALL_STAGING = YES
PENNYLANE_INSTALL_TARGET = YES

PENNYLANE_DEPENDENCIES = host-cmake python3 python-numpy

PENNYLANE_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTS=OFF \
	-DENABLE_OPENMP=OFF \
	-DENABLE_SCIPY_OPENBLAS=OFF

$(eval $(cmake-package))
