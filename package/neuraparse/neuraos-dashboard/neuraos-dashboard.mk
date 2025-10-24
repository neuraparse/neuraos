################################################################################
#
# neuraos-dashboard
#
################################################################################

NEURAOS_DASHBOARD_VERSION = 1.0.0
NEURAOS_DASHBOARD_SITE = $(BR2_EXTERNAL_NEURAOS_PATH)/src/dashboard
NEURAOS_DASHBOARD_SITE_METHOD = local
NEURAOS_DASHBOARD_DEPENDENCIES = qt5base qt5charts npie
NEURAOS_DASHBOARD_INSTALL_STAGING = NO
NEURAOS_DASHBOARD_INSTALL_TARGET = YES

$(eval $(cmake-package))

