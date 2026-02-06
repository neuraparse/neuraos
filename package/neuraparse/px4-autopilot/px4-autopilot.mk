################################################################################
#
# px4-autopilot - Professional Autopilot for Drones
#
################################################################################

# PX4 Autopilot v1.16.0 (2026) - Latest stable release
PX4_AUTOPILOT_VERSION = 1.16.0
PX4_AUTOPILOT_SITE = $(call github,PX4,PX4-Autopilot,v$(PX4_AUTOPILOT_VERSION))
PX4_AUTOPILOT_LICENSE = BSD-3-Clause
PX4_AUTOPILOT_LICENSE_FILES = LICENSE
PX4_AUTOPILOT_INSTALL_STAGING = YES
PX4_AUTOPILOT_INSTALL_TARGET = YES

PX4_AUTOPILOT_DEPENDENCIES = host-cmake mavlink

PX4_AUTOPILOT_CONF_OPTS = \
	-DCONFIG=px4_sitl_default \
	-DCMAKE_BUILD_TYPE=Release \
	-DENABLE_LOCKSTEP_SCHEDULER=ON

# Enable ROS2 integration
ifeq ($(BR2_PACKAGE_PX4_AUTOPILOT_ROS2),y)
PX4_AUTOPILOT_CONF_OPTS += -DROS2=ON
PX4_AUTOPILOT_DEPENDENCIES += fast-dds
endif

# Enable simulation
ifeq ($(BR2_PACKAGE_PX4_AUTOPILOT_SIMULATION),y)
PX4_AUTOPILOT_CONF_OPTS += -DENABLE_SIMULATION=ON
endif

$(eval $(cmake-package))
