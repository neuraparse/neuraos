#include "systeminfo.h"
#include <QFile>
#include <QTextStream>
#include <QStorageInfo>
#include <QSysInfo>
#include <QProcess>

SystemInfo::SystemInfo(QObject *parent) : QObject(parent)
{
    m_hostname = QSysInfo::machineHostName();
    m_kernelVersion = QSysInfo::kernelVersion();

    connect(&m_timer, &QTimer::timeout, this, &SystemInfo::refresh);
    m_timer.start(1000);
    refresh();
}

QString SystemInfo::hostname() const { return m_hostname; }
QString SystemInfo::kernelVersion() const { return m_kernelVersion; }

QString SystemInfo::uptime() const
{
    QFile f("/proc/uptime");
    if (f.open(QIODevice::ReadOnly)) {
        double secs = QString(f.readAll()).split(' ').first().toDouble();
        int h = (int)secs / 3600;
        int m = ((int)secs % 3600) / 60;
        int s = (int)secs % 60;
        return QString("%1h %2m %3s").arg(h).arg(m).arg(s);
    }
    return "N/A";
}

double SystemInfo::cpuUsage() const { return m_cpuUsage; }
qint64 SystemInfo::memoryUsed() const { return m_memUsed; }
qint64 SystemInfo::memoryTotal() const { return m_memTotal; }
qint64 SystemInfo::diskUsed() const { return m_diskUsed; }
qint64 SystemInfo::diskTotal() const { return m_diskTotal; }
double SystemInfo::cpuTemp() const { return m_cpuTemp; }
QVariantList SystemInfo::cpuHistory() const { return m_cpuHistory; }
QVariantList SystemInfo::memHistory() const { return m_memHistory; }

void SystemInfo::refresh()
{
    readCpuUsage();
    readMemory();
    readDisk();
    readTemp();

    /* Keep last 60 samples */
    m_cpuHistory.append(m_cpuUsage);
    if (m_cpuHistory.size() > 60) m_cpuHistory.removeFirst();

    double memPct = m_memTotal > 0 ? (double)m_memUsed / m_memTotal * 100.0 : 0;
    m_memHistory.append(memPct);
    if (m_memHistory.size() > 60) m_memHistory.removeFirst();

    emit updated();
}

void SystemInfo::readCpuUsage()
{
    QFile f("/proc/stat");
    if (!f.open(QIODevice::ReadOnly)) return;

    QString line = f.readLine();
    QStringList parts = line.simplified().split(' ');
    if (parts.size() < 8) return;

    long long user = parts[1].toLongLong();
    long long nice = parts[2].toLongLong();
    long long system = parts[3].toLongLong();
    long long idle = parts[4].toLongLong();
    long long iowait = parts[5].toLongLong();
    long long irq = parts[6].toLongLong();
    long long softirq = parts[7].toLongLong();

    long long total = user + nice + system + idle + iowait + irq + softirq;
    long long totalDiff = total - m_prevTotal;
    long long idleDiff = idle - m_prevIdle;

    if (totalDiff > 0) {
        m_cpuUsage = 100.0 * (1.0 - (double)idleDiff / totalDiff);
    }

    m_prevTotal = total;
    m_prevIdle = idle;
}

void SystemInfo::readMemory()
{
    QFile f("/proc/meminfo");
    if (!f.open(QIODevice::ReadOnly)) return;

    qint64 total = 0, available = 0;
    while (!f.atEnd()) {
        QString line = f.readLine();
        if (line.startsWith("MemTotal:"))
            total = line.split(':').last().trimmed().split(' ').first().toLongLong() * 1024;
        else if (line.startsWith("MemAvailable:"))
            available = line.split(':').last().trimmed().split(' ').first().toLongLong() * 1024;
    }
    m_memTotal = total;
    m_memUsed = total - available;
}

void SystemInfo::readDisk()
{
    QStorageInfo root = QStorageInfo::root();
    m_diskTotal = root.bytesTotal();
    m_diskUsed = root.bytesTotal() - root.bytesAvailable();
}

void SystemInfo::readTemp()
{
    QFile f("/sys/class/thermal/thermal_zone0/temp");
    if (f.open(QIODevice::ReadOnly)) {
        m_cpuTemp = f.readAll().trimmed().toDouble() / 1000.0;
    }
}
