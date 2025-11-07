################################################################################
#
# opencv-minimal
#
################################################################################

OPENCV_MINIMAL_VERSION = 4.12.0
OPENCV_MINIMAL_SITE = $(call github,opencv,opencv,$(OPENCV_MINIMAL_VERSION))
OPENCV_MINIMAL_LICENSE = Apache-2.0
OPENCV_MINIMAL_LICENSE_FILES = LICENSE
OPENCV_MINIMAL_INSTALL_STAGING = YES
OPENCV_MINIMAL_INSTALL_TARGET = YES

OPENCV_MINIMAL_DEPENDENCIES = host-cmake host-pkgconf zlib

# Minimal build configuration
OPENCV_MINIMAL_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DBUILD_opencv_apps=OFF \
	-DBUILD_opencv_calib3d=OFF \
	-DBUILD_opencv_features2d=OFF \
	-DBUILD_opencv_flann=OFF \
	-DBUILD_opencv_highgui=OFF \
	-DBUILD_opencv_ml=OFF \
	-DBUILD_opencv_objdetect=OFF \
	-DBUILD_opencv_photo=OFF \
	-DBUILD_opencv_stitching=OFF \
	-DBUILD_opencv_video=OFF \
	-DBUILD_TESTS=OFF \
	-DBUILD_PERF_TESTS=OFF \
	-DBUILD_EXAMPLES=OFF \
	-DBUILD_DOCS=OFF \
	-DWITH_GTK=OFF \
	-DWITH_QT=OFF \
	-DWITH_OPENGL=OFF \
	-DWITH_CUDA=OFF \
	-DWITH_OPENCL=OFF \
	-DWITH_IPP=OFF \
	-DWITH_TBB=OFF \
	-DWITH_EIGEN=OFF \
	-DWITH_V4L=ON \
	-DWITH_FFMPEG=OFF \
	-DWITH_GSTREAMER=OFF

# Enable core modules
OPENCV_MINIMAL_CONF_OPTS += \
	-DBUILD_opencv_core=ON \
	-DBUILD_opencv_imgproc=ON \
	-DBUILD_opencv_imgcodecs=ON

# Enable DNN module if selected
ifeq ($(BR2_PACKAGE_OPENCV_MINIMAL_DNN),y)
OPENCV_MINIMAL_CONF_OPTS += -DBUILD_opencv_dnn=ON
else
OPENCV_MINIMAL_CONF_OPTS += -DBUILD_opencv_dnn=OFF
endif

# Enable VideoIO if selected
ifeq ($(BR2_PACKAGE_OPENCV_MINIMAL_VIDEOIO),y)
OPENCV_MINIMAL_CONF_OPTS += -DBUILD_opencv_videoio=ON
else
OPENCV_MINIMAL_CONF_OPTS += -DBUILD_opencv_videoio=OFF
endif

# Enable NEON for ARM
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
OPENCV_MINIMAL_CONF_OPTS += -DENABLE_NEON=ON
endif

$(eval $(cmake-package))

