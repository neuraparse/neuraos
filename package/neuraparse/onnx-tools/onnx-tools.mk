################################################################################
#
# onnx-tools
#
################################################################################

ONNX_TOOLS_VERSION = 1.18.0
ONNX_TOOLS_SITE = $(call github,onnx,onnx,v$(ONNX_TOOLS_VERSION))
ONNX_TOOLS_LICENSE = Apache-2.0
ONNX_TOOLS_LICENSE_FILES = LICENSE
ONNX_TOOLS_SETUP_TYPE = setuptools
ONNX_TOOLS_INSTALL_STAGING = YES
ONNX_TOOLS_INSTALL_TARGET = YES

ONNX_TOOLS_DEPENDENCIES = host-python3 python3 python-numpy protobuf

$(eval $(python-package))

