################################################################################
#
# model-optimizer (NeuralOS AI Model Optimizer)
#
################################################################################

MODEL_OPTIMIZER_VERSION = 1.0.0
MODEL_OPTIMIZER_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/src/model-optimizer
MODEL_OPTIMIZER_SITE_METHOD = local
MODEL_OPTIMIZER_LICENSE = GPL-2.0-only
MODEL_OPTIMIZER_LICENSE_FILES = LICENSE
MODEL_OPTIMIZER_SETUP_TYPE = setuptools
MODEL_OPTIMIZER_INSTALL_STAGING = YES
MODEL_OPTIMIZER_INSTALL_TARGET = YES

MODEL_OPTIMIZER_DEPENDENCIES = host-python3 python3 python-numpy onnx-tools

$(eval $(python-package))

