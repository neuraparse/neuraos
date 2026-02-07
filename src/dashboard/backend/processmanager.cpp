#include "processmanager.h"
#include <QDir>
#include <QFile>
#include <QVariantMap>
#include <signal.h>

ProcessManager::ProcessManager(QObject *parent) : QObject(parent)
{
    refresh();
}

QVariantList ProcessManager::processes() const { return m_processes; }
int ProcessManager::processCount() const { return m_processes.size(); }

void ProcessManager::refresh()
{
    m_processes.clear();
    QDir proc("/proc");

    for (const QString &entry : proc.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        bool ok;
        int pid = entry.toInt(&ok);
        if (!ok) continue;

        QVariantMap p;
        p["pid"] = pid;

        /* Read comm */
        QFile comm(QString("/proc/%1/comm").arg(pid));
        if (comm.open(QIODevice::ReadOnly)) {
            p["name"] = QString(comm.readAll()).trimmed();
        } else {
            p["name"] = "?";
        }

        /* Read stat for state and memory */
        QFile stat(QString("/proc/%1/stat").arg(pid));
        if (stat.open(QIODevice::ReadOnly)) {
            QStringList fields = QString(stat.readAll()).split(' ');
            if (fields.size() > 23) {
                p["state"] = fields[2];
                p["vsize"] = fields[22].toLongLong() / 1024; /* KB */
                p["rss"] = fields[23].toLongLong() * 4; /* pages to KB */
            }
        }

        /* Read cmdline */
        QFile cmdline(QString("/proc/%1/cmdline").arg(pid));
        if (cmdline.open(QIODevice::ReadOnly)) {
            QString cmd = QString(cmdline.readAll()).replace('\0', ' ').trimmed();
            p["cmdline"] = cmd.left(80);
        }

        m_processes.append(p);
    }

    emit updated();
}

bool ProcessManager::killProcess(int pid, int signal)
{
    if (pid <= 1) return false;
    int ret = ::kill(pid, signal);
    if (ret == 0) {
        refresh();
        return true;
    }
    return false;
}
