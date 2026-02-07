#include "aibusmanager.h"
#include <QElapsedTimer>
#include <QRandomGenerator>

AIBusManager::AIBusManager(QObject *parent) : QObject(parent)
{
    /* Simulate periodic metrics updates */
    connect(&m_timer, &QTimer::timeout, this, &AIBusManager::simulateMetrics);
    m_timer.start(2000);

    /* Create default agents */
    createAgent("Vision Model", "mobilenet_v3", "litert");
    createAgent("Language Model", "llama-3.2-1b", "onnx");
    createAgent("Audio Classifier", "yamnet", "litert");
    createAgent("Anomaly Detector", "isolation_forest", "emlearn");

    /* Create a default pipeline */
    createPipeline("Perception Pipeline", {1, 3});
    createPipeline("Multimodal Analysis", {1, 2, 4});
}

int AIBusManager::agentCount() const { return m_agents.size(); }

int AIBusManager::activeAgents() const
{
    int count = 0;
    for (const auto &a : m_agents)
        if (a.status == "running") count++;
    return count;
}

int AIBusManager::pipelineCount() const { return m_pipelines.size(); }
quint64 AIBusManager::totalInferences() const { return m_totalInferences; }
double AIBusManager::busLatencyMs() const { return m_busLatencyMs; }

QVariantList AIBusManager::agents() const
{
    QVariantList list;
    for (const auto &a : m_agents) {
        QVariantMap m;
        m["id"] = a.id;
        m["name"] = a.name;
        m["model"] = a.model;
        m["backend"] = a.backend;
        m["status"] = a.status;
        m["memoryMB"] = a.memoryMB;
        m["inferences"] = (quint64)a.inferences;
        list.append(m);
    }
    return list;
}

QVariantList AIBusManager::pipelines() const
{
    QVariantList list;
    for (const auto &p : m_pipelines) {
        QVariantMap m;
        m["id"] = p.id;
        m["name"] = p.name;
        m["status"] = p.status;
        m["lastRunMs"] = p.lastRunMs;
        QVariantList ids;
        for (int id : p.agentIds) ids.append(id);
        m["agentIds"] = ids;
        list.append(m);
    }
    return list;
}

int AIBusManager::createAgent(const QString &name, const QString &model, const QString &backend)
{
    Agent a;
    a.id = m_nextAgentId++;
    a.name = name;
    a.model = model;
    a.backend = backend;
    a.status = "idle";
    a.memoryMB = QRandomGenerator::global()->bounded(20, 256);
    a.inferences = 0;
    m_agents.append(a);
    emit agentsChanged();
    return a.id;
}

bool AIBusManager::removeAgent(int id)
{
    for (int i = 0; i < m_agents.size(); i++) {
        if (m_agents[i].id == id) {
            m_agents.removeAt(i);
            emit agentsChanged();
            return true;
        }
    }
    return false;
}

bool AIBusManager::startAgent(int id)
{
    for (auto &a : m_agents) {
        if (a.id == id) {
            a.status = "running";
            emit agentsChanged();
            return true;
        }
    }
    return false;
}

bool AIBusManager::stopAgent(int id)
{
    for (auto &a : m_agents) {
        if (a.id == id) {
            a.status = "idle";
            emit agentsChanged();
            return true;
        }
    }
    return false;
}

QVariantMap AIBusManager::getAgentStatus(int id)
{
    for (const auto &a : m_agents) {
        if (a.id == id) {
            QVariantMap m;
            m["id"] = a.id;
            m["name"] = a.name;
            m["model"] = a.model;
            m["backend"] = a.backend;
            m["status"] = a.status;
            m["memoryMB"] = a.memoryMB;
            m["inferences"] = (quint64)a.inferences;
            return m;
        }
    }
    return {};
}

int AIBusManager::createPipeline(const QString &name, const QVariantList &agentIds)
{
    Pipeline p;
    p.id = m_nextPipelineId++;
    p.name = name;
    for (const auto &v : agentIds) p.agentIds.append(v.toInt());
    p.status = "idle";
    p.lastRunMs = 0;
    m_pipelines.append(p);
    emit pipelinesChanged();
    return p.id;
}

bool AIBusManager::removePipeline(int id)
{
    for (int i = 0; i < m_pipelines.size(); i++) {
        if (m_pipelines[i].id == id) {
            m_pipelines.removeAt(i);
            emit pipelinesChanged();
            return true;
        }
    }
    return false;
}

bool AIBusManager::executePipeline(int id)
{
    for (auto &p : m_pipelines) {
        if (p.id == id) {
            QElapsedTimer timer;
            timer.start();
            p.status = "running";
            emit pipelinesChanged();

            /* Simulate pipeline execution */
            for (int agentId : p.agentIds) {
                for (auto &a : m_agents) {
                    if (a.id == agentId) {
                        a.inferences++;
                        m_totalInferences++;
                    }
                }
            }

            p.lastRunMs = timer.nsecsElapsed() / 1000000.0 + QRandomGenerator::global()->bounded(5, 50);
            p.status = "completed";
            m_busLatencyMs = p.lastRunMs;
            emit pipelinesChanged();
            emit statsUpdated();
            return true;
        }
    }
    return false;
}

QVariantMap AIBusManager::getPipelineStatus(int id)
{
    for (const auto &p : m_pipelines) {
        if (p.id == id) {
            QVariantMap m;
            m["id"] = p.id;
            m["name"] = p.name;
            m["status"] = p.status;
            m["lastRunMs"] = p.lastRunMs;
            return m;
        }
    }
    return {};
}

void AIBusManager::refresh()
{
    simulateMetrics();
}

void AIBusManager::simulateMetrics()
{
    auto *rng = QRandomGenerator::global();
    for (auto &a : m_agents) {
        if (a.status == "running") {
            a.inferences += rng->bounded(1, 10);
            a.memoryMB += rng->bounded(-5, 5);
            if (a.memoryMB < 10) a.memoryMB = 10;
            m_totalInferences += rng->bounded(1, 10);
        }
    }
    m_busLatencyMs = rng->bounded(1, 45) + rng->generateDouble() * 10;
    emit agentsChanged();
    emit statsUpdated();
}
