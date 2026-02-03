################################################################################
#
# ncnn
#
################################################################################

# ncnn 20260114 (14 Jan 2026) - Tencent mobile AI framework
NCNN_VERSION = 20260114
NCNN_SITE = $(call github,Tencent,ncnn,$(NCNN_VERSION))
NCNN_LICENSE = BSD-3-Clause
NCNN_LICENSE_FILES = LICENSE.txt
NCNN_INSTALL_STAGING = YES
NCNN_INSTALL_TARGET = YES

NCNN_DEPENDENCIES = host-cmake protobuf

NCNN_CONF_OPTS = \
	-DNCNN_BUILD_TESTS=OFF \
	-DNCNN_BUILD_BENCHMARK=OFF \
	-DNCNN_BUILD_TOOLS=OFF \
	-DNCNN_BUILD_EXAMPLES=OFF \
	-DNCNN_SHARED_LIB=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable Vulkan support
ifeq ($(BR2_PACKAGE_NCNN_VULKAN),y)
NCNN_CONF_OPTS += -DNCNN_VULKAN=ON
else
NCNN_CONF_OPTS += -DNCNN_VULKAN=OFF
endif

# Enable OpenMP support
ifeq ($(BR2_PACKAGE_NCNN_OPENMP),y)
NCNN_CONF_OPTS += -DNCNN_OPENMP=ON
else
NCNN_CONF_OPTS += -DNCNN_OPENMP=OFF
endif

# Enable INT8 quantization
ifeq ($(BR2_PACKAGE_NCNN_INT8),y)
NCNN_CONF_OPTS += -DNCNN_INT8=ON
else
NCNN_CONF_OPTS += -DNCNN_INT8=OFF
endif

$(eval $(cmake-package))

