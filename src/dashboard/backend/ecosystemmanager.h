#ifndef ECOSYSTEMMANAGER_H
#define ECOSYSTEMMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QTimer>

/**
 * Device Ecosystem Manager (CosmOS + Fuchsia-inspired)
 * Multi-device management and AI task distribution
 */
class EcosystemManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int connectedCount READ connectedCount NOTIFY devicesChanged)
    Q_PROPERTY(QString syncStatus READ syncStatus NOTIFY syncChanged)
    Q_PROPERTY(QVariantList devices READ devices NOTIFY devicesChanged)
    Q_PROPERTY(int totalDevices READ totalDevices NOTIFY devicesChanged)

public:
    explicit EcosystemManager(QObject *parent = nullptr);

    int connectedCount() const;
    QString syncStatus() const;
    QVariantList devices() const;
    int totalDevices() const;

    Q_INVOKABLE void scanDevices();
    Q_INVOKABLE bool connectDevice(const QString &ip);
    Q_INVOKABLE bool disconnectDevice(int id);
    Q_INVOKABLE bool distributeTask(const QString &taskName, int deviceId);
    Q_INVOKABLE bool syncModels(int deviceId);
    Q_INVOKABLE QVariantMap getDeviceDetail(int id);
    Q_INVOKABLE void refresh();

signals:
    void devicesChanged();
    void syncChanged();
    void deviceDiscovered(const QString &name, const QString &ip);
    void taskDistributed(const QString &taskName, int deviceId);

private:
    struct Device {
        int id;
        QString name;
        QString ip;
        QString type; // "edge", "gateway", "sensor", "compute"
        QString status; // "connected", "disconnected", "syncing"
        QString arch; // "arm64", "x86_64", "riscv"
        double cpuUsage;
        double memoryUsage;
        int loadedModels;
        int activeTasks;
    };

    QList<Device> m_devices;
    QTimer m_timer;
    QString m_syncStatus = "idle";
    int m_nextId = 1;

    void loadDefaults();
    void simulateMetrics();
};

#endif
