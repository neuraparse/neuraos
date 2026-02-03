################################################################################
#
# llama-cpp
#
################################################################################

# llama.cpp b7746 (15 Jan 2026) - Latest with KleidiAI, Router Mode, improved backends
LLAMA_CPP_VERSION = b7746
LLAMA_CPP_SITE = $(call github,ggml-org,llama.cpp,$(LLAMA_CPP_VERSION))
LLAMA_CPP_LICENSE = MIT
LLAMA_CPP_LICENSE_FILES = LICENSE
LLAMA_CPP_INSTALL_STAGING = YES
LLAMA_CPP_INSTALL_TARGET = YES

LLAMA_CPP_DEPENDENCIES = host-cmake

LLAMA_CPP_CONF_OPTS = \
	-DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=ON \
	-DLLAMA_BUILD_SERVER=OFF \
	-DCMAKE_BUILD_TYPE=Release

# Enable ARM NEON
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_ARM_NEON=ON
endif

# Enable OpenCL backend
ifeq ($(BR2_PACKAGE_LLAMA_CPP_OPENCL),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_CLBLAST=ON
endif

# Enable Vulkan backend
ifeq ($(BR2_PACKAGE_LLAMA_CPP_VULKAN),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_VULKAN=ON
endif

# Enable Metal backend
ifeq ($(BR2_PACKAGE_LLAMA_CPP_METAL),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_METAL=ON
endif

$(eval $(cmake-package))

