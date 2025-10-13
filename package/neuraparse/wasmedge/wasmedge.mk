################################################################################
#
# wasmedge - WebAssembly Runtime for Edge Computing
#
################################################################################

WASMEDGE_VERSION = 0.14.1
WASMEDGE_SITE = $(call github,WasmEdge,WasmEdge,$(WASMEDGE_VERSION))
WASMEDGE_LICENSE = Apache-2.0
WASMEDGE_LICENSE_FILES = LICENSE
WASMEDGE_INSTALL_STAGING = YES
WASMEDGE_INSTALL_TARGET = YES

WASMEDGE_DEPENDENCIES = host-cmake llvm

# Build with CMake
WASMEDGE_CONF_OPTS = \
	-DWASMEDGE_BUILD_TESTS=OFF \
	-DWASMEDGE_BUILD_TOOLS=ON \
	-DWASMEDGE_BUILD_PLUGINS=ON \
	-DWASMEDGE_PLUGIN_WASI_NN_BACKEND=ON \
	-DCMAKE_BUILD_TYPE=Release

# Enable WASI-NN for AI inference
ifeq ($(BR2_PACKAGE_WASMEDGE_WASI_NN),y)
WASMEDGE_CONF_OPTS += -DWASMEDGE_PLUGIN_WASI_NN_BACKEND=ON
endif

$(eval $(cmake-package))

