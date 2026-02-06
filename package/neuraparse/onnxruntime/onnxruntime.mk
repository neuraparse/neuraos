################################################################################
#
# onnxruntime
#
################################################################################

ONNXRUNTIME_VERSION = 1.23.2
ONNXRUNTIME_SITE = $(call github,microsoft,onnxruntime,v$(ONNXRUNTIME_VERSION))
ONNXRUNTIME_LICENSE = MIT
ONNXRUNTIME_LICENSE_FILES = LICENSE
ONNXRUNTIME_INSTALL_STAGING = YES
ONNXRUNTIME_INSTALL_TARGET = YES

ONNXRUNTIME_DEPENDENCIES = host-cmake host-python3 protobuf flatbuffers

# ONNX Runtime has CMakeLists.txt in cmake/ subdirectory
ONNXRUNTIME_SUBDIR = cmake

# Directory for pre-downloaded dependencies
ONNXRUNTIME_DEPS_DIR = $(@D)/_deps

# Pre-configure: download external dependencies using wget (bypasses CMake's broken curl)
define ONNXRUNTIME_PRE_CONFIGURE_DOWNLOAD
	chmod +x $(ONNXRUNTIME_PKGDIR)/download-deps.sh
	$(ONNXRUNTIME_PKGDIR)/download-deps.sh $(ONNXRUNTIME_DEPS_DIR)
endef
ONNXRUNTIME_PRE_CONFIGURE_HOOKS += ONNXRUNTIME_PRE_CONFIGURE_DOWNLOAD

# Build with CMake - use pre-downloaded sources (FETCHCONTENT_FULLY_DISCONNECTED mode)
ONNXRUNTIME_CONF_OPTS = \
	-DONNX_CUSTOM_PROTOC_EXECUTABLE=$(HOST_DIR)/bin/protoc \
	-Donnxruntime_BUILD_SHARED_LIB=ON \
	-Donnxruntime_BUILD_UNIT_TESTS=OFF \
	-Donnxruntime_ENABLE_CPUINFO=ON \
	-Donnxruntime_USE_FULL_PROTOBUF=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DFETCHCONTENT_FULLY_DISCONNECTED=ON \
	-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
	-DFETCHCONTENT_SOURCE_DIR_ABSEIL_CPP=$(ONNXRUNTIME_DEPS_DIR)/abseil_cpp-src \
	-DFETCHCONTENT_SOURCE_DIR_EIGEN=$(ONNXRUNTIME_DEPS_DIR)/eigen-src \
	-DFETCHCONTENT_SOURCE_DIR_CPUINFO=$(ONNXRUNTIME_DEPS_DIR)/cpuinfo-src \
	-DFETCHCONTENT_SOURCE_DIR_ONNX=$(ONNXRUNTIME_DEPS_DIR)/onnx-src \
	-DFETCHCONTENT_SOURCE_DIR_SAFEINT=$(ONNXRUNTIME_DEPS_DIR)/safeint-src \
	-DFETCHCONTENT_SOURCE_DIR_GSL=$(ONNXRUNTIME_DEPS_DIR)/gsl-src \
	-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$(ONNXRUNTIME_DEPS_DIR)/nlohmann_json-src \
	-DFETCHCONTENT_SOURCE_DIR_DATE=$(ONNXRUNTIME_DEPS_DIR)/date-src \
	-DFETCHCONTENT_SOURCE_DIR_MP11=$(ONNXRUNTIME_DEPS_DIR)/mp11-src \
	-DFETCHCONTENT_SOURCE_DIR_RE2=$(ONNXRUNTIME_DEPS_DIR)/re2-src \
	-DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=$(ONNXRUNTIME_DEPS_DIR)/flatbuffers-src \
	-DFETCHCONTENT_SOURCE_DIR_PROTOBUF=$(ONNXRUNTIME_DEPS_DIR)/protobuf-src

# Enable OpenMP for multi-threading
ifeq ($(BR2_TOOLCHAIN_HAS_OPENMP),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_OPENMP=ON
endif

# Enable NEON for ARM
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_NEON=ON
endif

# Enable OpenVINO execution provider
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_OPENVINO),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_OPENVINO=ON
ONNXRUNTIME_DEPENDENCIES += openvino
endif

# Enable TensorRT execution provider
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_TENSORRT),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_TENSORRT=ON
ONNXRUNTIME_DEPENDENCIES += tensorrt
endif

# Enable ACL (ARM Compute Library) for Mali GPU
ifeq ($(BR2_PACKAGE_ONNXRUNTIME_ACL),y)
ONNXRUNTIME_CONF_OPTS += -Donnxruntime_USE_ACL=ON
ONNXRUNTIME_DEPENDENCIES += arm-compute-library
endif

$(eval $(cmake-package))
