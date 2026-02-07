#ifndef MCPMANAGER_H
#define MCPMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QDateTime>

/**
 * MCP Integration Hub Manager (Archon OS-inspired)
 * Model Context Protocol server for external AI assistant connectivity
 */
class MCPManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int connectedClients READ connectedClients NOTIFY clientsChanged)
    Q_PROPERTY(QString serverStatus READ serverStatus NOTIFY statusChanged)
    Q_PROPERTY(int serverPort READ serverPort NOTIFY statusChanged)
    Q_PROPERTY(QVariantList clients READ clients NOTIFY clientsChanged)
    Q_PROPERTY(QVariantList tools READ tools NOTIFY toolsChanged)
    Q_PROPERTY(QVariantList logs READ logs NOTIFY logsChanged)

public:
    explicit MCPManager(QObject *parent = nullptr);

    int connectedClients() const;
    QString serverStatus() const;
    int serverPort() const;
    QVariantList clients() const;
    QVariantList tools() const;
    QVariantList logs() const;

    Q_INVOKABLE bool startServer(int port = 3100);
    Q_INVOKABLE bool stopServer();
    Q_INVOKABLE void registerTool(const QString &name, const QString &description, const QString &schema);
    Q_INVOKABLE void unregisterTool(const QString &name);
    Q_INVOKABLE QVariantList getConnectedClients();
    Q_INVOKABLE void clearLogs();

signals:
    void clientsChanged();
    void statusChanged();
    void toolsChanged();
    void logsChanged();
    void clientConnected(const QString &name);
    void clientDisconnected(const QString &name);

private:
    struct Client {
        QString name;
        QString type; // "claude_code", "cursor", "windsurf", "other"
        QString status;
        QDateTime connectedAt;
        int requestCount;
    };

    struct Tool {
        QString name;
        QString description;
        QString schema;
        bool enabled;
        int callCount;
    };

    QList<Client> m_clients;
    QList<Tool> m_tools;
    QStringList m_logs;
    QString m_serverStatus = "stopped";
    int m_serverPort = 3100;

    void registerDefaultTools();
    void addLog(const QString &message);
};

#endif
