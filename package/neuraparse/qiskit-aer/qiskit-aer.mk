################################################################################
#
# qiskit-aer
#
################################################################################

# Qiskit Aer 0.17.2 (2026) - Latest quantum simulator
QISKIT_AER_VERSION = 0.17.2
QISKIT_AER_SITE = $(call github,Qiskit,qiskit-aer,$(QISKIT_AER_VERSION))
QISKIT_AER_LICENSE = Apache-2.0
QISKIT_AER_LICENSE_FILES = LICENSE.txt
QISKIT_AER_INSTALL_STAGING = YES
QISKIT_AER_INSTALL_TARGET = YES

QISKIT_AER_DEPENDENCIES = host-cmake python3 python-numpy

QISKIT_AER_CONF_OPTS = \
	-DBUILD_TESTS=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DAER_THRUST_BACKEND=OMP \
	-DAER_MPI=OFF

$(eval $(cmake-package))
