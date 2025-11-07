################################################################################
#
# apache-tvm
#
################################################################################

APACHE_TVM_VERSION = 0.22.0
APACHE_TVM_SITE = $(call github,apache,tvm,v$(APACHE_TVM_VERSION))
APACHE_TVM_LICENSE = Apache-2.0
APACHE_TVM_LICENSE_FILES = LICENSE
APACHE_TVM_INSTALL_STAGING = YES
APACHE_TVM_INSTALL_TARGET = YES

APACHE_TVM_DEPENDENCIES = host-cmake host-python3 llvm

APACHE_TVM_CONF_OPTS = \
	-DUSE_LLVM=ON \
	-DUSE_RELAY_DEBUG=OFF \
	-DUSE_GRAPH_EXECUTOR=ON \
	-DUSE_PROFILER=ON \
	-DBUILD_STATIC_RUNTIME=OFF \
	-DCMAKE_BUILD_TYPE=Release

# Enable LLVM backend
ifeq ($(BR2_PACKAGE_APACHE_TVM_LLVM_BACKEND),y)
APACHE_TVM_CONF_OPTS += -DUSE_LLVM=ON
endif

# Enable OpenCL backend
ifeq ($(BR2_PACKAGE_APACHE_TVM_OPENCL_BACKEND),y)
APACHE_TVM_CONF_OPTS += -DUSE_OPENCL=ON
endif

# Enable Vulkan backend
ifeq ($(BR2_PACKAGE_APACHE_TVM_VULKAN_BACKEND),y)
APACHE_TVM_CONF_OPTS += -DUSE_VULKAN=ON
endif

$(eval $(cmake-package))

