#ifndef SETTINGSMANAGER_H
#define SETTINGSMANAGER_H

#include <QObject>
#include <QSettings>

class SettingsManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int brightness READ brightness WRITE setBrightness NOTIFY changed)
    Q_PROPERTY(int volume READ volume WRITE setVolume NOTIFY changed)
    Q_PROPERTY(QString timezone READ timezone WRITE setTimezone NOTIFY changed)
    Q_PROPERTY(QString aiBackend READ aiBackend WRITE setAiBackend NOTIFY changed)
    Q_PROPERTY(QString theme READ theme WRITE setTheme NOTIFY changed)
    Q_PROPERTY(bool autoStart READ autoStart WRITE setAutoStart NOTIFY changed)

public:
    explicit SettingsManager(QObject *parent = nullptr);

    int brightness() const;
    void setBrightness(int v);
    int volume() const;
    void setVolume(int v);
    QString timezone() const;
    void setTimezone(const QString &v);
    QString aiBackend() const;
    void setAiBackend(const QString &v);
    QString theme() const;
    void setTheme(const QString &v);
    bool autoStart() const;
    void setAutoStart(bool v);

    Q_INVOKABLE void save();
    Q_INVOKABLE void load();
    Q_INVOKABLE void reset();

signals:
    void changed();

private:
    QSettings m_settings;
    int m_brightness = 80;
    int m_volume = 50;
    QString m_timezone = "UTC";
    QString m_aiBackend = "auto";
    QString m_theme = "dark";
    bool m_autoStart = true;
};

#endif
