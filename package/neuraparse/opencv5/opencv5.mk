################################################################################
#
# opencv5 - OpenCV 5.0 Computer Vision Library
#
################################################################################

# OpenCV 5.0.0-alpha (2026) - Next generation computer vision
OPENCV5_VERSION = 5.0.0-alpha
OPENCV5_SITE = $(call github,opencv,opencv,$(OPENCV5_VERSION))
OPENCV5_LICENSE = Apache-2.0
OPENCV5_LICENSE_FILES = LICENSE
OPENCV5_INSTALL_STAGING = YES
OPENCV5_INSTALL_TARGET = YES

OPENCV5_DEPENDENCIES = host-cmake zlib

# Core configuration for embedded systems
OPENCV5_CONF_OPTS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DBUILD_opencv_apps=OFF \
	-DBUILD_DOCS=OFF \
	-DBUILD_EXAMPLES=OFF \
	-DBUILD_TESTS=OFF \
	-DBUILD_PERF_TESTS=OFF \
	-DWITH_GTK=OFF \
	-DWITH_QT=OFF \
	-DWITH_FFMPEG=OFF \
	-DWITH_V4L=ON \
	-DWITH_OPENCL=OFF \
	-DWITH_CUDA=OFF \
	-DBUILD_LIST=core,imgproc,imgcodecs,videoio,calib3d,features2d,objdetect,dnn

# Enable NEON optimizations on ARM
ifeq ($(BR2_ARM_CPU_HAS_NEON),y)
OPENCV5_CONF_OPTS += -DENABLE_NEON=ON
endif

# Enable ARM SIMD via KleidiCV
ifeq ($(BR2_aarch64),y)
OPENCV5_CONF_OPTS += -DWITH_KLEIDICV=ON
endif

# Enable DNN module
ifeq ($(BR2_PACKAGE_OPENCV5_DNN),y)
OPENCV5_CONF_OPTS += -DBUILD_opencv_dnn=ON
else
OPENCV5_CONF_OPTS += -DBUILD_opencv_dnn=OFF
endif

# Enable Python bindings
ifeq ($(BR2_PACKAGE_OPENCV5_PYTHON),y)
OPENCV5_DEPENDENCIES += python3
OPENCV5_CONF_OPTS += -DBUILD_opencv_python3=ON
else
OPENCV5_CONF_OPTS += -DBUILD_opencv_python3=OFF
endif

$(eval $(cmake-package))
