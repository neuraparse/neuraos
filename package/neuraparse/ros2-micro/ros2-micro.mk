################################################################################
#
# ros2-micro
#
################################################################################

# micro-ROS Agent iron (ROS 2 Iron compatible)
ROS2_MICRO_VERSION = iron
ROS2_MICRO_SITE = $(call github,micro-ROS,micro_ros_setup,$(ROS2_MICRO_VERSION))
ROS2_MICRO_LICENSE = Apache-2.0
ROS2_MICRO_LICENSE_FILES = LICENSE
ROS2_MICRO_INSTALL_TARGET = YES

ROS2_MICRO_DEPENDENCIES = host-cmake

# Install marker for now (full ROS2 requires complex setup)
define ROS2_MICRO_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/share/ros2-micro
	echo "micro-ROS $(ROS2_MICRO_VERSION)" > $(TARGET_DIR)/usr/share/ros2-micro/version
endef

$(eval $(generic-package))
