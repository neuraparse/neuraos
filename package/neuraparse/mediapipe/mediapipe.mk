################################################################################
#
# mediapipe
#
################################################################################

MEDIAPIPE_VERSION = 0.10.26
MEDIAPIPE_SITE = $(call github,google-ai-edge,mediapipe,v$(MEDIAPIPE_VERSION))
MEDIAPIPE_LICENSE = Apache-2.0
MEDIAPIPE_LICENSE_FILES = LICENSE
MEDIAPIPE_INSTALL_STAGING = YES
MEDIAPIPE_INSTALL_TARGET = YES

MEDIAPIPE_DEPENDENCIES = host-cmake opencv-minimal protobuf

MEDIAPIPE_CONF_OPTS = \
	-DMEDIAPIPE_DISABLE_GPU=ON \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable GPU support
ifeq ($(BR2_PACKAGE_MEDIAPIPE_GPU),y)
MEDIAPIPE_CONF_OPTS += -DMEDIAPIPE_DISABLE_GPU=OFF
endif

$(eval $(cmake-package))

