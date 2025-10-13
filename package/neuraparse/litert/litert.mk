################################################################################
#
# litert (formerly TensorFlow Lite)
#
################################################################################

LITERT_VERSION = 2.18.0
LITERT_SITE = $(call github,google-ai-edge,LiteRT,v$(LITERT_VERSION))
LITERT_LICENSE = Apache-2.0
LITERT_LICENSE_FILES = LICENSE
LITERT_INSTALL_STAGING = YES
LITERT_INSTALL_TARGET = YES

LITERT_DEPENDENCIES = host-cmake host-python3 flatbuffers

# Build with CMake
LITERT_CONF_OPTS = \
	-DTFLITE_ENABLE_XNNPACK=ON \
	-DTFLITE_ENABLE_GPU=ON \
	-DTFLITE_ENABLE_NNAPI=ON \
	-DTFLITE_ENABLE_RUY=ON \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable NEON for ARM
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
LITERT_CONF_OPTS += -DTFLITE_ENABLE_NEON=ON
endif

# Enable GPU delegate
ifeq ($(BR2_PACKAGE_LITERT_GPU_DELEGATE),y)
LITERT_CONF_OPTS += -DTFLITE_ENABLE_GPU=ON
LITERT_DEPENDENCIES += mesa3d libdrm
endif

# Enable XNNPACK delegate (optimized for ARM/x86)
ifeq ($(BR2_PACKAGE_LITERT_XNNPACK),y)
LITERT_CONF_OPTS += -DTFLITE_ENABLE_XNNPACK=ON
endif

# Enable NNAPI delegate (for Android/NPU)
ifeq ($(BR2_PACKAGE_LITERT_NNAPI_DELEGATE),y)
LITERT_CONF_OPTS += -DTFLITE_ENABLE_NNAPI=ON
endif

$(eval $(cmake-package))

