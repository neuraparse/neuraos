################################################################################
#
# foonathan-memory - Memory allocator for Fast-DDS
#
################################################################################

# foonathan_memory v0.7-3 (stable release) - direct library, not vendor wrapper
FOONATHAN_MEMORY_VERSION = v0.7-3
FOONATHAN_MEMORY_SITE = $(call github,foonathan,memory,$(FOONATHAN_MEMORY_VERSION))
FOONATHAN_MEMORY_LICENSE = Zlib
FOONATHAN_MEMORY_LICENSE_FILES = LICENSE
FOONATHAN_MEMORY_INSTALL_STAGING = YES
FOONATHAN_MEMORY_INSTALL_TARGET = YES

FOONATHAN_MEMORY_DEPENDENCIES = host-cmake

FOONATHAN_MEMORY_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DFOONATHAN_MEMORY_BUILD_EXAMPLES=OFF \
	-DFOONATHAN_MEMORY_BUILD_TESTS=OFF \
	-DFOONATHAN_MEMORY_BUILD_TOOLS=ON

$(eval $(cmake-package))
