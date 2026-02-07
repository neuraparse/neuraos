#include "networkmanager.h"
#include <QDir>
#include <QFile>
#include <QVariantMap>
#include <QSysInfo>
#include <QNetworkInterface>

NetworkManager::NetworkManager(QObject *parent) : QObject(parent)
{
    m_hostname = QSysInfo::machineHostName();
    refresh();
}

QVariantList NetworkManager::interfaces() const { return m_interfaces; }
QString NetworkManager::hostname() const { return m_hostname; }

void NetworkManager::refresh()
{
    m_interfaces.clear();

    for (const QNetworkInterface &iface : QNetworkInterface::allInterfaces()) {
        if (iface.flags() & QNetworkInterface::IsLoopBack) continue;

        QVariantMap info;
        info["name"] = iface.name();
        info["hwAddress"] = iface.hardwareAddress();
        info["isUp"] = (bool)(iface.flags() & QNetworkInterface::IsUp);
        info["isRunning"] = (bool)(iface.flags() & QNetworkInterface::IsRunning);

        QStringList addrs;
        for (const QNetworkAddressEntry &addr : iface.addressEntries()) {
            addrs << addr.ip().toString();
        }
        info["addresses"] = addrs.join(", ");

        /* Read traffic stats from /proc/net/dev */
        QFile f("/proc/net/dev");
        if (f.open(QIODevice::ReadOnly)) {
            while (!f.atEnd()) {
                QString line = f.readLine().trimmed();
                if (line.startsWith(iface.name() + ":")) {
                    QStringList parts = line.split(':').last().simplified().split(' ');
                    if (parts.size() >= 10) {
                        info["rxBytes"] = parts[0].toLongLong();
                        info["rxPackets"] = parts[1].toLongLong();
                        info["txBytes"] = parts[8].toLongLong();
                        info["txPackets"] = parts[9].toLongLong();
                    }
                    break;
                }
            }
        }

        m_interfaces.append(info);
    }

    emit updated();
}
