#include "automationmanager.h"

AutomationManager::AutomationManager(QObject *parent) : QObject(parent)
{
    loadDefaults();
}

QVariantList AutomationManager::workflows() const
{
    QVariantList list;
    for (const auto &w : m_workflows) {
        QVariantMap m;
        m["id"] = w.id;
        m["name"] = w.name;
        m["description"] = w.description;
        m["status"] = w.status;
        m["stepCount"] = w.steps.size();
        m["created"] = w.created.toString("yyyy-MM-dd hh:mm");
        m["lastRun"] = w.lastRun.isValid() ? w.lastRun.toString("yyyy-MM-dd hh:mm") : "Never";
        m["runCount"] = w.runCount;
        list.append(m);
    }
    return list;
}

bool AutomationManager::isRecording() const { return m_isRecording; }
int AutomationManager::activeWorkflow() const { return m_activeWorkflow; }
int AutomationManager::workflowCount() const { return m_workflows.size(); }

int AutomationManager::createWorkflow(const QString &name, const QString &description)
{
    Workflow w;
    w.id = m_nextId++;
    w.name = name;
    w.description = description;
    w.status = "idle";
    w.created = QDateTime::currentDateTime();
    w.runCount = 0;
    m_workflows.append(w);
    emit workflowsChanged();
    return w.id;
}

bool AutomationManager::deleteWorkflow(int id)
{
    for (int i = 0; i < m_workflows.size(); i++) {
        if (m_workflows[i].id == id) {
            m_workflows.removeAt(i);
            emit workflowsChanged();
            return true;
        }
    }
    return false;
}

void AutomationManager::startRecording(int workflowId)
{
    for (auto &w : m_workflows) {
        if (w.id == workflowId) {
            w.status = "recording";
            m_isRecording = true;
            m_activeWorkflow = workflowId;
            emit recordingChanged();
            emit activeChanged();
            emit workflowsChanged();
            return;
        }
    }
}

void AutomationManager::stopRecording()
{
    for (auto &w : m_workflows) {
        if (w.id == m_activeWorkflow) {
            w.status = "idle";
        }
    }
    m_isRecording = false;
    m_activeWorkflow = -1;
    emit recordingChanged();
    emit activeChanged();
    emit workflowsChanged();
}

bool AutomationManager::addStep(int workflowId, const QString &type, const QVariantMap &params)
{
    for (auto &w : m_workflows) {
        if (w.id == workflowId) {
            Step s;
            s.type = type;
            s.params = params;
            w.steps.append(s);
            emit workflowsChanged();
            return true;
        }
    }
    return false;
}

bool AutomationManager::removeStep(int workflowId, int stepIndex)
{
    for (auto &w : m_workflows) {
        if (w.id == workflowId && stepIndex >= 0 && stepIndex < w.steps.size()) {
            w.steps.removeAt(stepIndex);
            emit workflowsChanged();
            return true;
        }
    }
    return false;
}

bool AutomationManager::playWorkflow(int id)
{
    for (auto &w : m_workflows) {
        if (w.id == id) {
            w.status = "running";
            m_activeWorkflow = id;
            emit activeChanged();
            emit workflowsChanged();

            /* Simulate step execution */
            for (int i = 0; i < w.steps.size(); i++) {
                emit stepExecuted(id, i);
            }

            w.status = "completed";
            w.lastRun = QDateTime::currentDateTime();
            w.runCount++;
            m_activeWorkflow = -1;
            emit activeChanged();
            emit workflowsChanged();
            emit workflowCompleted(id, true);
            return true;
        }
    }
    return false;
}

bool AutomationManager::stopWorkflow(int id)
{
    for (auto &w : m_workflows) {
        if (w.id == id) {
            w.status = "idle";
            m_activeWorkflow = -1;
            emit activeChanged();
            emit workflowsChanged();
            return true;
        }
    }
    return false;
}

QVariantMap AutomationManager::getWorkflowDetail(int id)
{
    for (const auto &w : m_workflows) {
        if (w.id == id) {
            QVariantMap m;
            m["id"] = w.id;
            m["name"] = w.name;
            m["description"] = w.description;
            m["status"] = w.status;
            m["created"] = w.created.toString("yyyy-MM-dd hh:mm:ss");
            m["lastRun"] = w.lastRun.isValid() ? w.lastRun.toString("yyyy-MM-dd hh:mm:ss") : "Never";
            m["runCount"] = w.runCount;

            QVariantList steps;
            for (const auto &s : w.steps) {
                QVariantMap sm;
                sm["type"] = s.type;
                sm["params"] = s.params;
                steps.append(sm);
            }
            m["steps"] = steps;
            return m;
        }
    }
    return {};
}

void AutomationManager::loadDefaults()
{
    /* Pre-built example workflows */
    int w1 = createWorkflow("Morning System Check", "Opens system monitor, checks NPU status, runs benchmark");
    addStep(w1, "open_app", {{"app", "SystemMonitorApp.qml"}});
    addStep(w1, "wait", {{"seconds", 2}});
    addStep(w1, "open_app", {{"app", "NPUControlCenter.qml"}});
    addStep(w1, "ai_inference", {{"model", "benchmark_model.tflite"}});

    int w2 = createWorkflow("Deploy Model Pipeline", "Loads model, runs test inference, logs results");
    addStep(w2, "open_app", {{"app", "NeuralStudio.qml"}});
    addStep(w2, "ai_inference", {{"model", "mobilenet_v3.tflite"}});
    addStep(w2, "wait", {{"seconds", 1}});
    addStep(w2, "type_text", {{"text", "Model deployed successfully"}});

    int w3 = createWorkflow("Security Audit", "Scans network, checks processes, reviews logs");
    addStep(w3, "open_app", {{"app", "NetworkCenter.qml"}});
    addStep(w3, "open_app", {{"app", "TaskManagerApp.qml"}});
    addStep(w3, "open_app", {{"app", "DefenseMonitor.qml"}});
}
