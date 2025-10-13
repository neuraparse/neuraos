################################################################################
#
# npie - NeuraParse Inference Engine
#
################################################################################

NPIE_VERSION = 1.0.0
NPIE_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/src/npie
NPIE_SITE_METHOD = local
NPIE_LICENSE = Proprietary
NPIE_LICENSE_FILES = LICENSE
NPIE_INSTALL_STAGING = YES
NPIE_INSTALL_TARGET = YES

NPIE_DEPENDENCIES = host-cmake

# Add LiteRT dependency if enabled
ifeq ($(BR2_PACKAGE_LITERT),y)
NPIE_DEPENDENCIES += litert
NPIE_CONF_OPTS += -DENABLE_LITERT=ON
else
NPIE_CONF_OPTS += -DENABLE_LITERT=OFF
endif

# Add ONNX Runtime dependency if enabled
ifeq ($(BR2_PACKAGE_ONNXRUNTIME),y)
NPIE_DEPENDENCIES += onnxruntime
NPIE_CONF_OPTS += -DENABLE_ONNXRUNTIME=ON
else
NPIE_CONF_OPTS += -DENABLE_ONNXRUNTIME=OFF
endif

# Add emlearn dependency if enabled
ifeq ($(BR2_PACKAGE_EMLEARN),y)
NPIE_DEPENDENCIES += emlearn
NPIE_CONF_OPTS += -DENABLE_EMLEARN=ON
else
NPIE_CONF_OPTS += -DENABLE_EMLEARN=OFF
endif

# Add OpenCV dependency if enabled
ifeq ($(BR2_PACKAGE_OPENCV_MINIMAL),y)
NPIE_DEPENDENCIES += opencv-minimal
NPIE_CONF_OPTS += -DENABLE_OPENCV=ON
else
NPIE_CONF_OPTS += -DENABLE_OPENCV=OFF
endif

# Add WasmEdge dependency if enabled
ifeq ($(BR2_PACKAGE_WASMEDGE),y)
NPIE_DEPENDENCIES += wasmedge
NPIE_CONF_OPTS += -DENABLE_WASMEDGE=ON
else
NPIE_CONF_OPTS += -DENABLE_WASMEDGE=OFF
endif

# Build configuration
NPIE_CONF_OPTS += \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DBUILD_NPIE=ON \
	-DBUILD_TOOLS=ON \
	-DBUILD_EXAMPLES=ON

# Install init script
define NPIE_INSTALL_INIT_SYSV
	$(INSTALL) -D -m 0755 $(BR2_EXTERNAL_NEURAOS_PATH)/package/neuraparse/npie/S90npie \
		$(TARGET_DIR)/etc/init.d/S90npie
endef

# Install configuration
define NPIE_INSTALL_CONFIG
	$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_NEURAOS_PATH)/package/neuraparse/npie/npie.conf \
		$(TARGET_DIR)/etc/npie/config.json
endef

NPIE_POST_INSTALL_TARGET_HOOKS += NPIE_INSTALL_INIT_SYSV
NPIE_POST_INSTALL_TARGET_HOOKS += NPIE_INSTALL_CONFIG

$(eval $(cmake-package))

