# NeuralOS (NeuraParse) Buildroot external makefile aggregator
# Include all package makefiles under package/neuraparse/*

include $(sort $(wildcard $(BR2_EXTERNAL_NEURAOS_PATH)/package/neuraparse/*/*.mk))

