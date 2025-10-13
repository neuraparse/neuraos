################################################################################
#
# onnxruntime
#
################################################################################

ONNXRUNTIME_VERSION = 1.20.1
ONNXRUNTIME_SITE = $(call github,microsoft,onnxruntime,v$(ONNXRUNTIME_VERSION))
ONNXRUNTIME_LICENSE = MIT
ONNXRUNTIME_LICENSE_FILES = LICENSE
ONNXRUNTIME_INSTALL_STAGING = YES
ONNXRUNTIME_INSTALL_TARGET = YES

ONNXRUNTIME_DEPENDENCIES = host-cmake host-python3 protobuf flatbuffers

# Build with CMake
ONNXRUNTIME_CONF_OPTS = \
	-DONNX_CUSTOM_PROTOC_EXECUTABLE=$(HOST_DIR)/bin/protoc \
	-Donnxruntime_BUILD_SHARED_LIB=ON \
	-Donnxruntime_BUILD_UNIT_TESTS=OFF \
	-Donnxruntime_USE_PREINSTALLED_PROTOBUF=ON \
	-Donnxruntime_ENABLE_CPUINFO=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable OpenMP for multi-threading
ifeq ($(BR2_TOOLCHAIN_HAS_OPENMP),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_OPENMP=ON
endif

# Enable NEON for ARM
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_NEON=ON
endif

# Enable OpenVINO execution provider
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_OPENVINO),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_OPENVINO=ON
ONNXRUNTIME_DEPENDENCIES += openvino
endif

# Enable TensorRT execution provider
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_TENSORRT),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_TENSORRT=ON
ONNXRUNTIME_DEPENDENCIES += tensorrt
endif

# Enable ACL (ARM Compute Library) for Mali GPU
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_ACL),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_ACL=ON
ONNXRUNTIME_DEPENDENCIES += arm-compute-library
endif

$(eval $(cmake-package))

