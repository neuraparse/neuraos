#include "npiebridge.h"
#include <QFileInfo>
#include <QElapsedTimer>

extern "C" {
#include "npie.h"
}

NPIEBridge::NPIEBridge(QObject *parent) : QObject(parent)
{
    npie_context_t ctx;
    npie_options_t opts = {};
    opts.backend = NPIE_BACKEND_AUTO;
    opts.accelerator = NPIE_ACCELERATOR_NONE;
    opts.num_threads = 1;
    opts.timeout_ms = 1000;

    if (npie_init(&ctx, &opts) == NPIE_SUCCESS) {
        m_ctx = (void*)(intptr_t)ctx;
    }
}

NPIEBridge::~NPIEBridge()
{
    if (m_ctx) {
        npie_shutdown((npie_context_t)(intptr_t)m_ctx);
    }
}

QString NPIEBridge::version() const
{
    return QString(npie_version());
}

QString NPIEBridge::currentBackend() const { return m_currentBackend; }

QStringList NPIEBridge::backends() const
{
    return {"auto", "litert", "onnx", "emlearn", "wasm"};
}

bool NPIEBridge::modelLoaded() const { return m_modelLoaded; }
QString NPIEBridge::modelName() const { return m_modelName; }
quint64 NPIEBridge::inferenceCount() const { return m_inferenceCount; }
double NPIEBridge::lastInferenceMs() const { return m_lastInferenceMs; }

double NPIEBridge::avgInferenceMs() const
{
    return m_inferenceCount > 0 ? m_totalInferenceMs / m_inferenceCount : 0;
}

bool NPIEBridge::loadModel(const QString &path)
{
    QFileInfo fi(path);
    m_modelName = fi.fileName();
    m_modelLoaded = true;
    m_inferenceCount = 0;
    m_totalInferenceMs = 0;
    m_lastInferenceMs = 0;
    emit modelChanged();
    return true;
}

void NPIEBridge::unloadModel()
{
    m_modelLoaded = false;
    m_modelName.clear();
    emit modelChanged();
}

double NPIEBridge::runInference()
{
    if (!m_modelLoaded) return -1;

    QElapsedTimer timer;
    timer.start();

    /* Simulate inference workload */
    volatile double sum = 0;
    for (int i = 0; i < 100000; i++) {
        sum += (double)i * 0.001;
    }

    m_lastInferenceMs = timer.nsecsElapsed() / 1000000.0;
    m_totalInferenceMs += m_lastInferenceMs;
    m_inferenceCount++;
    emit statsUpdated();
    return m_lastInferenceMs;
}

bool NPIEBridge::setBackend(const QString &name)
{
    if (backends().contains(name)) {
        m_currentBackend = name;
        emit backendChanged();
        return true;
    }
    return false;
}
