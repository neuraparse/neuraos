################################################################################
#
# openvino - Intel OpenVINO Toolkit for Edge AI
#
################################################################################

# OpenVINO 2024.6 (2026) - Intel AI inference toolkit
OPENVINO_VERSION = 2024.6.0
OPENVINO_SITE = $(call github,openvinotoolkit,openvino,$(OPENVINO_VERSION))
OPENVINO_LICENSE = Apache-2.0
OPENVINO_LICENSE_FILES = LICENSE
OPENVINO_INSTALL_STAGING = YES
OPENVINO_INSTALL_TARGET = YES

OPENVINO_DEPENDENCIES = host-cmake protobuf

# Minimal embedded configuration
OPENVINO_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DENABLE_INTEL_CPU=ON \
	-DENABLE_INTEL_GPU=OFF \
	-DENABLE_INTEL_NPU=OFF \
	-DENABLE_TESTS=OFF \
	-DENABLE_SAMPLES=OFF \
	-DENABLE_DOCS=OFF \
	-DENABLE_PYTHON=OFF \
	-DENABLE_WHEEL=OFF \
	-DENABLE_OV_ONNX_FRONTEND=ON \
	-DENABLE_OV_PADDLE_FRONTEND=OFF \
	-DENABLE_OV_TF_FRONTEND=ON \
	-DENABLE_OV_TF_LITE_FRONTEND=ON \
	-DBUILD_SHARED_LIBS=ON \
	-DENABLE_SYSTEM_PROTOBUF=ON

# Enable ARM optimizations
ifeq ($(BR2_aarch64),y)
OPENVINO_CONF_OPTS += -DENABLE_ARM_COMPUTE_CMAKE=ON
endif

# Enable Neural Network Compiler (for NPU)
ifeq ($(BR2_PACKAGE_OPENVINO_NPU),y)
OPENVINO_CONF_OPTS += -DENABLE_INTEL_NPU=ON
endif

# Enable Python bindings
ifeq ($(BR2_PACKAGE_OPENVINO_PYTHON),y)
OPENVINO_DEPENDENCIES += python3 python-numpy
OPENVINO_CONF_OPTS += -DENABLE_PYTHON=ON
endif

$(eval $(cmake-package))
