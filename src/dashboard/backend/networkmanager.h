#ifndef NETWORKMANAGER_H
#define NETWORKMANAGER_H

#include <QObject>
#include <QVariantList>

class NetworkManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList interfaces READ interfaces NOTIFY updated)
    Q_PROPERTY(QString hostname READ hostname CONSTANT)

public:
    explicit NetworkManager(QObject *parent = nullptr);

    QVariantList interfaces() const;
    QString hostname() const;

    Q_INVOKABLE void refresh();

signals:
    void updated();

private:
    QVariantList m_interfaces;
    QString m_hostname;
};

#endif
