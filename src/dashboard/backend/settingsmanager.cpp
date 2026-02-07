#include "settingsmanager.h"

SettingsManager::SettingsManager(QObject *parent)
    : QObject(parent)
    , m_settings("/etc/neuraos/settings.ini", QSettings::IniFormat)
{
    load();
}

int SettingsManager::brightness() const { return m_brightness; }
void SettingsManager::setBrightness(int v) { m_brightness = v; emit changed(); }
int SettingsManager::volume() const { return m_volume; }
void SettingsManager::setVolume(int v) { m_volume = v; emit changed(); }
QString SettingsManager::timezone() const { return m_timezone; }
void SettingsManager::setTimezone(const QString &v) { m_timezone = v; emit changed(); }
QString SettingsManager::aiBackend() const { return m_aiBackend; }
void SettingsManager::setAiBackend(const QString &v) { m_aiBackend = v; emit changed(); }
QString SettingsManager::theme() const { return m_theme; }
void SettingsManager::setTheme(const QString &v) { m_theme = v; emit changed(); }
bool SettingsManager::autoStart() const { return m_autoStart; }
void SettingsManager::setAutoStart(bool v) { m_autoStart = v; emit changed(); }

void SettingsManager::save()
{
    m_settings.setValue("display/brightness", m_brightness);
    m_settings.setValue("audio/volume", m_volume);
    m_settings.setValue("system/timezone", m_timezone);
    m_settings.setValue("ai/backend", m_aiBackend);
    m_settings.setValue("ui/theme", m_theme);
    m_settings.setValue("system/autostart", m_autoStart);
    m_settings.sync();
}

void SettingsManager::load()
{
    m_brightness = m_settings.value("display/brightness", 80).toInt();
    m_volume = m_settings.value("audio/volume", 50).toInt();
    m_timezone = m_settings.value("system/timezone", "UTC").toString();
    m_aiBackend = m_settings.value("ai/backend", "auto").toString();
    m_theme = m_settings.value("ui/theme", "dark").toString();
    m_autoStart = m_settings.value("system/autostart", true).toBool();
    emit changed();
}

void SettingsManager::reset()
{
    m_brightness = 80;
    m_volume = 50;
    m_timezone = "UTC";
    m_aiBackend = "auto";
    m_theme = "dark";
    m_autoStart = true;
    save();
}
