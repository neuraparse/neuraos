################################################################################
#
# neuraos-tests
#
################################################################################

NEURAOS_TESTS_VERSION = 1.0.0
NEURAOS_TESTS_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/tests
NEURAOS_TESTS_SITE_METHOD = local
NEURAOS_TESTS_LICENSE = GPL-2.0-only

define NEURAOS_TESTS_BUILD_CMDS
	$(TARGET_CC) $(TARGET_CFLAGS) $(TARGET_LDFLAGS) \
		$(@D)/test_ai_stack.c -o $(@D)/test_ai_stack
endef

define NEURAOS_TESTS_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/test_ai_stack $(TARGET_DIR)/usr/bin/test_ai_stack
endef

$(eval $(generic-package))

