################################################################################
#
# llama-cpp
#
################################################################################

# llama.cpp b7907 (February 2026) - Latest with improved performance
LLAMA_CPP_VERSION = b7907
LLAMA_CPP_SITE = $(call github,ggml-org,llama.cpp,$(LLAMA_CPP_VERSION))
LLAMA_CPP_LICENSE = MIT
LLAMA_CPP_LICENSE_FILES = LICENSE
LLAMA_CPP_INSTALL_STAGING = YES
LLAMA_CPP_INSTALL_TARGET = YES

LLAMA_CPP_DEPENDENCIES = host-cmake

LLAMA_CPP_CONF_OPTS = \
	-DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=ON \
	-DLLAMA_BUILD_SERVER=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable x86 optimizations
ifeq ($(BR2_x86_64),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_AVX2=ON -DLLAMA_AVX512=OFF
endif

# Enable ARM NEON
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
LLAMA_CPP_CONF_OPTS += -DLLAMA_ARM_NEON=ON
endif

$(eval $(cmake-package))
