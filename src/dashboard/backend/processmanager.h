#ifndef PROCESSMANAGER_H
#define PROCESSMANAGER_H

#include <QObject>
#include <QVariantList>

class ProcessManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList processes READ processes NOTIFY updated)
    Q_PROPERTY(int processCount READ processCount NOTIFY updated)

public:
    explicit ProcessManager(QObject *parent = nullptr);

    QVariantList processes() const;
    int processCount() const;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE bool killProcess(int pid, int signal = 15);

signals:
    void updated();

private:
    QVariantList m_processes;
};

#endif
