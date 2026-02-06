################################################################################
#
# piper-tts
#
################################################################################

# Piper 2023.11.14-2 (Latest stable release)
PIPER_TTS_VERSION = 2023.11.14-2
PIPER_TTS_SITE = $(call github,rhasspy,piper,$(PIPER_TTS_VERSION))
PIPER_TTS_LICENSE = MIT
PIPER_TTS_LICENSE_FILES = LICENSE.md
PIPER_TTS_INSTALL_STAGING = YES
PIPER_TTS_INSTALL_TARGET = YES

PIPER_TTS_DEPENDENCIES = host-cmake

PIPER_TTS_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTING=OFF

$(eval $(cmake-package))
