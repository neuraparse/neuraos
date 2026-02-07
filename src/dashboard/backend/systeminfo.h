#ifndef SYSTEMINFO_H
#define SYSTEMINFO_H

#include <QObject>
#include <QTimer>
#include <QVariantList>

class SystemInfo : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString hostname READ hostname NOTIFY updated)
    Q_PROPERTY(QString kernelVersion READ kernelVersion NOTIFY updated)
    Q_PROPERTY(QString uptime READ uptime NOTIFY updated)
    Q_PROPERTY(double cpuUsage READ cpuUsage NOTIFY updated)
    Q_PROPERTY(qint64 memoryUsed READ memoryUsed NOTIFY updated)
    Q_PROPERTY(qint64 memoryTotal READ memoryTotal NOTIFY updated)
    Q_PROPERTY(qint64 diskUsed READ diskUsed NOTIFY updated)
    Q_PROPERTY(qint64 diskTotal READ diskTotal NOTIFY updated)
    Q_PROPERTY(double cpuTemp READ cpuTemp NOTIFY updated)
    Q_PROPERTY(QVariantList cpuHistory READ cpuHistory NOTIFY updated)
    Q_PROPERTY(QVariantList memHistory READ memHistory NOTIFY updated)

public:
    explicit SystemInfo(QObject *parent = nullptr);

    QString hostname() const;
    QString kernelVersion() const;
    QString uptime() const;
    double cpuUsage() const;
    qint64 memoryUsed() const;
    qint64 memoryTotal() const;
    qint64 diskUsed() const;
    qint64 diskTotal() const;
    double cpuTemp() const;
    QVariantList cpuHistory() const;
    QVariantList memHistory() const;

    Q_INVOKABLE void refresh();

signals:
    void updated();

private:
    void readCpuUsage();
    void readMemory();
    void readDisk();
    void readTemp();

    QTimer m_timer;
    QString m_hostname;
    QString m_kernelVersion;
    double m_cpuUsage = 0;
    qint64 m_memUsed = 0;
    qint64 m_memTotal = 0;
    qint64 m_diskUsed = 0;
    qint64 m_diskTotal = 0;
    double m_cpuTemp = 0;
    QVariantList m_cpuHistory;
    QVariantList m_memHistory;

    /* For CPU delta calculation */
    long long m_prevIdle = 0;
    long long m_prevTotal = 0;
};

#endif
