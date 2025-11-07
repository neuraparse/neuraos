################################################################################
#
# orb-slam3
#
################################################################################

ORB_SLAM3_VERSION = 1.0
ORB_SLAM3_SITE = $(call github,UZ-SLAMLab,ORB_SLAM3,v$(ORB_SLAM3_VERSION))
ORB_SLAM3_LICENSE = GPL-3.0
ORB_SLAM3_LICENSE_FILES = LICENSE
ORB_SLAM3_INSTALL_STAGING = YES
ORB_SLAM3_INSTALL_TARGET = YES

ORB_SLAM3_DEPENDENCIES = host-cmake opencv-minimal eigen

ORB_SLAM3_CONF_OPTS = \
	-DBUILD_EXAMPLES=OFF \
	-DCMAKE_BUILD_TYPE=Release

# Enable Pangolin viewer
ifeq ($(BR2_PACKAGE_ORB_SLAM3_VIEWER),y)
ORB_SLAM3_CONF_OPTS += -DUSE_PANGOLIN=ON
ORB_SLAM3_DEPENDENCIES += pangolin
else
ORB_SLAM3_CONF_OPTS += -DUSE_PANGOLIN=OFF
endif

$(eval $(cmake-package))

