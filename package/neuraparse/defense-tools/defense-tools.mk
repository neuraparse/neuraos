################################################################################
#
# defense-tools
#
################################################################################

# Meta-package for defense and security tools
DEFENSE_TOOLS_VERSION = 1.0
DEFENSE_TOOLS_LICENSE = MIT
DEFENSE_TOOLS_SITE_METHOD = local
DEFENSE_TOOLS_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/package/neuraparse/defense-tools

# This is a meta-package that just pulls in dependencies
# based on configuration options

ifeq ($(BR2_PACKAGE_DEFENSE_TOOLS_NETWORK),y)
DEFENSE_TOOLS_DEPENDENCIES += tcpdump iptables
endif

ifeq ($(BR2_PACKAGE_DEFENSE_TOOLS_CRYPTO),y)
DEFENSE_TOOLS_DEPENDENCIES += openssl
endif

ifeq ($(BR2_PACKAGE_DEFENSE_TOOLS_AUDIT),y)
DEFENSE_TOOLS_DEPENDENCIES += audit
endif

# Create a marker file to indicate installation
define DEFENSE_TOOLS_INSTALL_TARGET_CMDS
	mkdir -p $(TARGET_DIR)/etc/neuraos
	echo "Defense Tools $(DEFENSE_TOOLS_VERSION)" > $(TARGET_DIR)/etc/neuraos/defense-tools.version
endef

$(eval $(generic-package))
