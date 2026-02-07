#include "mcpmanager.h"
#include <QDateTime>

MCPManager::MCPManager(QObject *parent) : QObject(parent)
{
    registerDefaultTools();
}

int MCPManager::connectedClients() const { return m_clients.size(); }
QString MCPManager::serverStatus() const { return m_serverStatus; }
int MCPManager::serverPort() const { return m_serverPort; }

QVariantList MCPManager::clients() const
{
    QVariantList list;
    for (const auto &c : m_clients) {
        QVariantMap m;
        m["name"] = c.name;
        m["type"] = c.type;
        m["status"] = c.status;
        m["connectedAt"] = c.connectedAt.toString("hh:mm:ss");
        m["requestCount"] = c.requestCount;
        list.append(m);
    }
    return list;
}

QVariantList MCPManager::tools() const
{
    QVariantList list;
    for (const auto &t : m_tools) {
        QVariantMap m;
        m["name"] = t.name;
        m["description"] = t.description;
        m["schema"] = t.schema;
        m["enabled"] = t.enabled;
        m["callCount"] = t.callCount;
        list.append(m);
    }
    return list;
}

QVariantList MCPManager::logs() const
{
    QVariantList list;
    for (const auto &l : m_logs)
        list.append(l);
    return list;
}

bool MCPManager::startServer(int port)
{
    m_serverPort = port;
    m_serverStatus = "running";
    addLog("MCP server started on port " + QString::number(port));

    /* Simulate some connected clients */
    Client c1;
    c1.name = "Claude Code";
    c1.type = "claude_code";
    c1.status = "connected";
    c1.connectedAt = QDateTime::currentDateTime();
    c1.requestCount = 0;
    m_clients.append(c1);

    Client c2;
    c2.name = "Cursor IDE";
    c2.type = "cursor";
    c2.status = "connected";
    c2.connectedAt = QDateTime::currentDateTime();
    c2.requestCount = 0;
    m_clients.append(c2);

    addLog("Client connected: Claude Code");
    addLog("Client connected: Cursor IDE");

    emit statusChanged();
    emit clientsChanged();
    return true;
}

bool MCPManager::stopServer()
{
    m_serverStatus = "stopped";
    m_clients.clear();
    addLog("MCP server stopped");
    emit statusChanged();
    emit clientsChanged();
    return true;
}

void MCPManager::registerTool(const QString &name, const QString &description, const QString &schema)
{
    Tool t;
    t.name = name;
    t.description = description;
    t.schema = schema;
    t.enabled = true;
    t.callCount = 0;
    m_tools.append(t);
    addLog("Tool registered: " + name);
    emit toolsChanged();
}

void MCPManager::unregisterTool(const QString &name)
{
    for (int i = 0; i < m_tools.size(); i++) {
        if (m_tools[i].name == name) {
            m_tools.removeAt(i);
            addLog("Tool unregistered: " + name);
            emit toolsChanged();
            return;
        }
    }
}

QVariantList MCPManager::getConnectedClients()
{
    return clients();
}

void MCPManager::clearLogs()
{
    m_logs.clear();
    emit logsChanged();
}

void MCPManager::registerDefaultTools()
{
    registerTool("system_info",
                 "Get system information (CPU, memory, disk, temperature)",
                 "{\"type\":\"object\",\"properties\":{\"metric\":{\"type\":\"string\",\"enum\":[\"cpu\",\"memory\",\"disk\",\"temperature\",\"all\"]}}}");

    registerTool("run_inference",
                 "Run AI model inference through NPIE",
                 "{\"type\":\"object\",\"properties\":{\"model\":{\"type\":\"string\"},\"backend\":{\"type\":\"string\",\"enum\":[\"auto\",\"litert\",\"onnx\",\"emlearn\",\"wasm\"]}},\"required\":[\"model\"]}");

    registerTool("file_browse",
                 "Browse and read files on the device",
                 "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"action\":{\"type\":\"string\",\"enum\":[\"list\",\"read\",\"search\"]}},\"required\":[\"path\"]}");

    registerTool("npu_status",
                 "Get NPU accelerator status and utilization",
                 "{\"type\":\"object\",\"properties\":{\"detail\":{\"type\":\"string\",\"enum\":[\"summary\",\"full\"]}}}");

    registerTool("process_list",
                 "List running processes with resource usage",
                 "{\"type\":\"object\",\"properties\":{\"sort\":{\"type\":\"string\",\"enum\":[\"cpu\",\"memory\",\"name\"]}}}");

    registerTool("ai_memory_query",
                 "Query the shared AI memory system",
                 "{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"recall\",\"store\",\"search\"]},\"key\":{\"type\":\"string\"},\"value\":{\"type\":\"string\"}},\"required\":[\"action\"]}");

    registerTool("network_info",
                 "Get network interfaces and connectivity status",
                 "{\"type\":\"object\",\"properties\":{}}");

    registerTool("ecosystem_devices",
                 "List connected edge devices in the ecosystem",
                 "{\"type\":\"object\",\"properties\":{}}");
}

void MCPManager::addLog(const QString &message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    m_logs.prepend("[" + timestamp + "] " + message);
    if (m_logs.size() > 100)
        m_logs = m_logs.mid(0, 100);
    emit logsChanged();
}
