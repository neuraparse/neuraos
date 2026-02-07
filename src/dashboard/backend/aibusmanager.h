#ifndef AIBUSMANAGER_H
#define AIBUSMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QTimer>

/**
 * AI Bus Orchestration Manager (CosmOS-inspired)
 * Coordinates multiple AI models and services through pipelines
 */
class AIBusManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int agentCount READ agentCount NOTIFY agentsChanged)
    Q_PROPERTY(int activeAgents READ activeAgents NOTIFY agentsChanged)
    Q_PROPERTY(int pipelineCount READ pipelineCount NOTIFY pipelinesChanged)
    Q_PROPERTY(quint64 totalInferences READ totalInferences NOTIFY statsUpdated)
    Q_PROPERTY(double busLatencyMs READ busLatencyMs NOTIFY statsUpdated)
    Q_PROPERTY(QVariantList agents READ agents NOTIFY agentsChanged)
    Q_PROPERTY(QVariantList pipelines READ pipelines NOTIFY pipelinesChanged)

public:
    explicit AIBusManager(QObject *parent = nullptr);

    int agentCount() const;
    int activeAgents() const;
    int pipelineCount() const;
    quint64 totalInferences() const;
    double busLatencyMs() const;
    QVariantList agents() const;
    QVariantList pipelines() const;

    Q_INVOKABLE int createAgent(const QString &name, const QString &model, const QString &backend);
    Q_INVOKABLE bool removeAgent(int id);
    Q_INVOKABLE bool startAgent(int id);
    Q_INVOKABLE bool stopAgent(int id);
    Q_INVOKABLE QVariantMap getAgentStatus(int id);

    Q_INVOKABLE int createPipeline(const QString &name, const QVariantList &agentIds);
    Q_INVOKABLE bool removePipeline(int id);
    Q_INVOKABLE bool executePipeline(int id);
    Q_INVOKABLE QVariantMap getPipelineStatus(int id);

    Q_INVOKABLE void refresh();

signals:
    void agentsChanged();
    void pipelinesChanged();
    void statsUpdated();

private:
    struct Agent {
        int id;
        QString name;
        QString model;
        QString backend;
        QString status; // "idle", "running", "error"
        double memoryMB;
        quint64 inferences;
    };

    struct Pipeline {
        int id;
        QString name;
        QList<int> agentIds;
        QString status; // "idle", "running", "completed", "error"
        double lastRunMs;
    };

    QList<Agent> m_agents;
    QList<Pipeline> m_pipelines;
    QTimer m_timer;
    int m_nextAgentId = 1;
    int m_nextPipelineId = 1;
    quint64 m_totalInferences = 0;
    double m_busLatencyMs = 0;

    void simulateMetrics();
};

#endif
