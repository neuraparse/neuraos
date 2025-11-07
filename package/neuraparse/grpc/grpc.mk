################################################################################
#
# grpc
#
################################################################################

GRPC_VERSION = 1.76.0
GRPC_SITE = $(call github,grpc,grpc,v$(GRPC_VERSION))
GRPC_LICENSE = Apache-2.0
GRPC_LICENSE_FILES = LICENSE
GRPC_INSTALL_STAGING = YES
GRPC_INSTALL_TARGET = YES

GRPC_DEPENDENCIES = host-cmake protobuf openssl zlib c-ares

GRPC_CONF_OPTS = \
	-DgRPC_BUILD_TESTS=OFF \
	-DgRPC_BUILD_CODEGEN=ON \
	-DgRPC_BUILD_CSHARP_EXT=OFF \
	-DgRPC_INSTALL=ON \
	-DgRPC_PROTOBUF_PROVIDER=package \
	-DgRPC_SSL_PROVIDER=package \
	-DgRPC_ZLIB_PROVIDER=package \
	-DgRPC_CARES_PROVIDER=package \
	-DCMAKE_BUILD_TYPE=Release

$(eval $(cmake-package))

