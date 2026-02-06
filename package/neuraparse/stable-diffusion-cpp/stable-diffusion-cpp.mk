################################################################################
#
# stable-diffusion-cpp
#
################################################################################

# stable-diffusion.cpp (February 2026) - SD/SDXL/Flux support
STABLE_DIFFUSION_CPP_VERSION = master-492-f957fa3
STABLE_DIFFUSION_CPP_SITE = $(call github,leejet,stable-diffusion.cpp,$(STABLE_DIFFUSION_CPP_VERSION))
STABLE_DIFFUSION_CPP_LICENSE = MIT
STABLE_DIFFUSION_CPP_LICENSE_FILES = LICENSE
STABLE_DIFFUSION_CPP_INSTALL_STAGING = YES
STABLE_DIFFUSION_CPP_INSTALL_TARGET = YES

STABLE_DIFFUSION_CPP_DEPENDENCIES = host-cmake

STABLE_DIFFUSION_CPP_CONF_OPTS = \
	-DSD_BUILD_EXAMPLES=ON \
	-DCMAKE_BUILD_TYPE=Release

$(eval $(cmake-package))
