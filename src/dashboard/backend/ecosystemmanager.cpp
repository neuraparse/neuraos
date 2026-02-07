#include "ecosystemmanager.h"
#include <QRandomGenerator>

EcosystemManager::EcosystemManager(QObject *parent) : QObject(parent)
{
    connect(&m_timer, &QTimer::timeout, this, &EcosystemManager::simulateMetrics);
    m_timer.start(3000);
    loadDefaults();
}

int EcosystemManager::connectedCount() const
{
    int count = 0;
    for (const auto &d : m_devices)
        if (d.status == "connected") count++;
    return count;
}

QString EcosystemManager::syncStatus() const { return m_syncStatus; }
int EcosystemManager::totalDevices() const { return m_devices.size(); }

QVariantList EcosystemManager::devices() const
{
    QVariantList list;
    for (const auto &d : m_devices) {
        QVariantMap m;
        m["id"] = d.id;
        m["name"] = d.name;
        m["ip"] = d.ip;
        m["type"] = d.type;
        m["status"] = d.status;
        m["arch"] = d.arch;
        m["cpuUsage"] = d.cpuUsage;
        m["memoryUsage"] = d.memoryUsage;
        m["loadedModels"] = d.loadedModels;
        m["activeTasks"] = d.activeTasks;
        list.append(m);
    }
    return list;
}

void EcosystemManager::scanDevices()
{
    m_syncStatus = "scanning";
    emit syncChanged();

    /* Simulate mDNS discovery */
    if (m_devices.size() < 8) {
        Device d;
        d.id = m_nextId++;
        d.name = "Discovered Node " + QString::number(d.id);
        d.ip = "192.168.1." + QString::number(50 + d.id);
        d.type = "sensor";
        d.status = "disconnected";
        d.arch = "arm64";
        d.cpuUsage = 0;
        d.memoryUsage = 0;
        d.loadedModels = 0;
        d.activeTasks = 0;
        m_devices.append(d);
        emit deviceDiscovered(d.name, d.ip);
    }

    m_syncStatus = "idle";
    emit syncChanged();
    emit devicesChanged();
}

bool EcosystemManager::connectDevice(const QString &ip)
{
    for (auto &d : m_devices) {
        if (d.ip == ip) {
            d.status = "connected";
            d.cpuUsage = QRandomGenerator::global()->bounded(10, 60);
            d.memoryUsage = QRandomGenerator::global()->bounded(20, 70);
            emit devicesChanged();
            return true;
        }
    }
    return false;
}

bool EcosystemManager::disconnectDevice(int id)
{
    for (auto &d : m_devices) {
        if (d.id == id) {
            d.status = "disconnected";
            d.cpuUsage = 0;
            d.memoryUsage = 0;
            d.activeTasks = 0;
            emit devicesChanged();
            return true;
        }
    }
    return false;
}

bool EcosystemManager::distributeTask(const QString &taskName, int deviceId)
{
    for (auto &d : m_devices) {
        if (d.id == deviceId && d.status == "connected") {
            d.activeTasks++;
            emit taskDistributed(taskName, deviceId);
            emit devicesChanged();
            return true;
        }
    }
    return false;
}

bool EcosystemManager::syncModels(int deviceId)
{
    for (auto &d : m_devices) {
        if (d.id == deviceId && d.status == "connected") {
            m_syncStatus = "syncing";
            emit syncChanged();

            d.loadedModels += 1;

            m_syncStatus = "idle";
            emit syncChanged();
            emit devicesChanged();
            return true;
        }
    }
    return false;
}

QVariantMap EcosystemManager::getDeviceDetail(int id)
{
    for (const auto &d : m_devices) {
        if (d.id == id) {
            QVariantMap m;
            m["id"] = d.id;
            m["name"] = d.name;
            m["ip"] = d.ip;
            m["type"] = d.type;
            m["status"] = d.status;
            m["arch"] = d.arch;
            m["cpuUsage"] = d.cpuUsage;
            m["memoryUsage"] = d.memoryUsage;
            m["loadedModels"] = d.loadedModels;
            m["activeTasks"] = d.activeTasks;
            return m;
        }
    }
    return {};
}

void EcosystemManager::refresh()
{
    simulateMetrics();
}

void EcosystemManager::simulateMetrics()
{
    auto *rng = QRandomGenerator::global();
    for (auto &d : m_devices) {
        if (d.status == "connected") {
            d.cpuUsage = qBound(5.0, d.cpuUsage + rng->bounded(-10, 10), 95.0);
            d.memoryUsage = qBound(10.0, d.memoryUsage + rng->bounded(-5, 5), 90.0);
        }
    }
    emit devicesChanged();
}

void EcosystemManager::loadDefaults()
{
    auto add = [this](const QString &name, const QString &ip, const QString &type,
                      const QString &arch, const QString &status) {
        Device d;
        d.id = m_nextId++;
        d.name = name;
        d.ip = ip;
        d.type = type;
        d.status = status;
        d.arch = arch;
        d.cpuUsage = status == "connected" ? QRandomGenerator::global()->bounded(10, 60) : 0;
        d.memoryUsage = status == "connected" ? QRandomGenerator::global()->bounded(20, 70) : 0;
        d.loadedModels = status == "connected" ? QRandomGenerator::global()->bounded(1, 4) : 0;
        d.activeTasks = status == "connected" ? QRandomGenerator::global()->bounded(0, 3) : 0;
        m_devices.append(d);
    };

    add("Edge Node Alpha",   "192.168.1.10", "compute",  "arm64",   "connected");
    add("Sensor Hub Beta",   "192.168.1.11", "sensor",   "arm64",   "connected");
    add("Gateway Gamma",     "192.168.1.12", "gateway",  "x86_64",  "connected");
    add("Drone Unit Delta",  "192.168.1.13", "edge",     "arm64",   "disconnected");
    add("RISC-V Dev Board",  "192.168.1.14", "compute",  "riscv",   "disconnected");

    emit devicesChanged();
}
