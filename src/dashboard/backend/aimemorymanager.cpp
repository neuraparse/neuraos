#include "aimemorymanager.h"

AIMemoryManager::AIMemoryManager(QObject *parent) : QObject(parent)
{
    loadDefaults();
}

int AIMemoryManager::memoryEntries() const { return m_entries.size(); }

int AIMemoryManager::totalSize() const
{
    int size = 0;
    for (const auto &e : m_entries)
        size += e.key.size() + e.value.size();
    return size;
}

QStringList AIMemoryManager::categories() const
{
    QStringList cats;
    for (const auto &e : m_entries) {
        if (!cats.contains(e.category))
            cats.append(e.category);
    }
    return cats;
}

QVariantList AIMemoryManager::entries() const
{
    QVariantList list;
    for (const auto &e : m_entries) {
        QVariantMap m;
        m["key"] = e.key;
        m["value"] = e.value;
        m["category"] = e.category;
        m["source"] = e.source;
        m["timestamp"] = e.timestamp.toString("yyyy-MM-dd hh:mm:ss");
        m["accessCount"] = e.accessCount;
        list.append(m);
    }
    return list;
}

bool AIMemoryManager::store(const QString &key, const QString &value, const QString &category, const QString &source)
{
    /* Update if exists */
    for (auto &e : m_entries) {
        if (e.key == key) {
            e.value = value;
            e.category = category;
            e.source = source;
            e.timestamp = QDateTime::currentDateTime();
            emit memoryChanged();
            return true;
        }
    }

    MemoryEntry entry;
    entry.key = key;
    entry.value = value;
    entry.category = category;
    entry.source = source;
    entry.timestamp = QDateTime::currentDateTime();
    entry.accessCount = 0;
    m_entries.append(entry);
    emit memoryChanged();
    return true;
}

QString AIMemoryManager::recall(const QString &key)
{
    for (auto &e : m_entries) {
        if (e.key == key) {
            e.accessCount++;
            emit memoryChanged();
            return e.value;
        }
    }
    return {};
}

QVariantList AIMemoryManager::search(const QString &query)
{
    QVariantList results;
    QString q = query.toLower();
    for (const auto &e : m_entries) {
        if (e.key.toLower().contains(q) || e.value.toLower().contains(q)) {
            QVariantMap m;
            m["key"] = e.key;
            m["value"] = e.value;
            m["category"] = e.category;
            m["source"] = e.source;
            m["timestamp"] = e.timestamp.toString("yyyy-MM-dd hh:mm:ss");
            results.append(m);
        }
    }
    return results;
}

QVariantList AIMemoryManager::getByCategory(const QString &category)
{
    QVariantList results;
    for (const auto &e : m_entries) {
        if (e.category == category) {
            QVariantMap m;
            m["key"] = e.key;
            m["value"] = e.value;
            m["source"] = e.source;
            m["timestamp"] = e.timestamp.toString("yyyy-MM-dd hh:mm:ss");
            results.append(m);
        }
    }
    return results;
}

QVariantMap AIMemoryManager::getContext()
{
    QVariantMap ctx;
    ctx["totalEntries"] = memoryEntries();
    ctx["totalSize"] = totalSize();
    ctx["categories"] = QVariant(categories());

    /* Latest 5 entries as recent context */
    QVariantList recent;
    int start = qMax(0, m_entries.size() - 5);
    for (int i = m_entries.size() - 1; i >= start; i--) {
        QVariantMap m;
        m["key"] = m_entries[i].key;
        m["value"] = m_entries[i].value;
        recent.append(m);
    }
    ctx["recent"] = recent;
    return ctx;
}

bool AIMemoryManager::removeEntry(const QString &key)
{
    for (int i = 0; i < m_entries.size(); i++) {
        if (m_entries[i].key == key) {
            m_entries.removeAt(i);
            emit memoryChanged();
            return true;
        }
    }
    return false;
}

void AIMemoryManager::clearCategory(const QString &category)
{
    for (int i = m_entries.size() - 1; i >= 0; i--) {
        if (m_entries[i].category == category)
            m_entries.removeAt(i);
    }
    emit memoryChanged();
}

void AIMemoryManager::clearAll()
{
    m_entries.clear();
    emit memoryChanged();
}

void AIMemoryManager::loadDefaults()
{
    store("user.preferred_backend", "litert", "preferences", "Settings");
    store("user.preferred_theme", "dark", "preferences", "Settings");
    store("last_model_loaded", "mobilenet_v3.tflite", "tasks", "Neural Studio");
    store("npu_performance_baseline", "4.55 GFLOPS", "knowledge", "NPU Control");
    store("system_boot_time", "4.2 seconds", "knowledge", "System Monitor");
    store("last_inference_result", "Classification: cat (92.3%)", "tasks", "AI Assistant");
    store("ai_context.conversation_topic", "NPU inference pipeline", "conversations", "AI Assistant");
    store("ai_context.user_expertise", "Advanced - Edge AI Developer", "conversations", "AI Assistant");
}
