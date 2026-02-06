################################################################################
#
# whisper-cpp
#
################################################################################

# whisper.cpp v1.8.3 (January 2026) - 12x iGPU performance boost
WHISPER_CPP_VERSION = v1.8.3
WHISPER_CPP_SITE = $(call github,ggml-org,whisper.cpp,$(WHISPER_CPP_VERSION))
WHISPER_CPP_LICENSE = MIT
WHISPER_CPP_LICENSE_FILES = LICENSE
WHISPER_CPP_INSTALL_STAGING = YES
WHISPER_CPP_INSTALL_TARGET = YES

WHISPER_CPP_DEPENDENCIES = host-cmake

WHISPER_CPP_CONF_OPTS = \
	-DWHISPER_BUILD_TESTS=OFF \
	-DWHISPER_BUILD_EXAMPLES=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable x86 optimizations
ifeq ($(BR2_x86_64),y)
WHISPER_CPP_CONF_OPTS += -DWHISPER_NO_AVX2=OFF
endif

# Enable ARM NEON
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
WHISPER_CPP_CONF_OPTS += -DWHISPER_NO_NEON=OFF
else
WHISPER_CPP_CONF_OPTS += -DWHISPER_NO_NEON=ON
endif

$(eval $(cmake-package))
