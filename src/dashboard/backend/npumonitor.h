#ifndef NPUMONITOR_H
#define NPUMONITOR_H

#include <QObject>
#include <QTimer>
#include <QVariantList>

class NPUMonitor : public QObject {
    Q_OBJECT
    Q_PROPERTY(int deviceCount READ deviceCount NOTIFY updated)
    Q_PROPERTY(QVariantList devices READ devices NOTIFY updated)
    Q_PROPERTY(quint64 totalInferences READ totalInferences NOTIFY updated)
    Q_PROPERTY(quint64 totalTimeUs READ totalTimeUs NOTIFY updated)
    Q_PROPERTY(quint64 powerMw READ powerMw NOTIFY updated)
    Q_PROPERTY(quint32 frequencyMhz READ frequencyMhz NOTIFY updated)
    Q_PROPERTY(bool powerEnabled READ powerEnabled NOTIFY updated)

public:
    explicit NPUMonitor(QObject *parent = nullptr);

    int deviceCount() const;
    QVariantList devices() const;
    quint64 totalInferences() const;
    quint64 totalTimeUs() const;
    quint64 powerMw() const;
    quint32 frequencyMhz() const;
    bool powerEnabled() const;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE bool setPower(bool enabled);
    Q_INVOKABLE bool setFrequency(quint32 mhz);

signals:
    void updated();

private:
    QTimer m_timer;
    int m_deviceCount = 0;
    QVariantList m_devices;
    quint64 m_totalInferences = 0;
    quint64 m_totalTimeUs = 0;
    quint64 m_powerMw = 0;
    quint32 m_frequencyMhz = 0;
    bool m_powerEnabled = false;
    bool m_initialized = false;
};

#endif
