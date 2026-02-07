#ifndef AUTOMATIONMANAGER_H
#define AUTOMATIONMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QDateTime>

/**
 * Visual Automation Studio Manager (WarmWind OS-inspired)
 * Record and replay workflow automations
 */
class AutomationManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList workflows READ workflows NOTIFY workflowsChanged)
    Q_PROPERTY(bool isRecording READ isRecording NOTIFY recordingChanged)
    Q_PROPERTY(int activeWorkflow READ activeWorkflow NOTIFY activeChanged)
    Q_PROPERTY(int workflowCount READ workflowCount NOTIFY workflowsChanged)

public:
    explicit AutomationManager(QObject *parent = nullptr);

    QVariantList workflows() const;
    bool isRecording() const;
    int activeWorkflow() const;
    int workflowCount() const;

    Q_INVOKABLE int createWorkflow(const QString &name, const QString &description);
    Q_INVOKABLE bool deleteWorkflow(int id);
    Q_INVOKABLE void startRecording(int workflowId);
    Q_INVOKABLE void stopRecording();
    Q_INVOKABLE bool addStep(int workflowId, const QString &type, const QVariantMap &params);
    Q_INVOKABLE bool removeStep(int workflowId, int stepIndex);
    Q_INVOKABLE bool playWorkflow(int id);
    Q_INVOKABLE bool stopWorkflow(int id);
    Q_INVOKABLE QVariantMap getWorkflowDetail(int id);

signals:
    void workflowsChanged();
    void recordingChanged();
    void activeChanged();
    void workflowCompleted(int id, bool success);
    void stepExecuted(int workflowId, int stepIndex);

private:
    struct Step {
        QString type; // "open_app", "click", "type_text", "wait", "condition", "ai_inference"
        QVariantMap params;
    };

    struct Workflow {
        int id;
        QString name;
        QString description;
        QList<Step> steps;
        QString status; // "idle", "recording", "running", "completed", "error"
        QDateTime created;
        QDateTime lastRun;
        int runCount;
    };

    QList<Workflow> m_workflows;
    bool m_isRecording = false;
    int m_activeWorkflow = -1;
    int m_nextId = 1;

    void loadDefaults();
};

#endif
