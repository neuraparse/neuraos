#include "knowledgemanager.h"
#include <QFileInfo>

KnowledgeManager::KnowledgeManager(QObject *parent) : QObject(parent)
{
    loadDefaults();
}

int KnowledgeManager::totalDocs() const { return m_documents.size(); }
QString KnowledgeManager::indexStatus() const { return m_indexStatus; }

QVariantList KnowledgeManager::documents() const
{
    QVariantList list;
    for (const auto &d : m_documents) {
        QVariantMap m;
        m["id"] = d.id;
        m["path"] = d.path;
        m["title"] = d.title;
        m["type"] = d.type;
        m["chunks"] = d.chunks;
        m["sizeBytes"] = d.sizeBytes;
        m["indexed"] = d.indexed.toString("yyyy-MM-dd hh:mm");
        list.append(m);
    }
    return list;
}

int KnowledgeManager::totalChunks() const
{
    int total = 0;
    for (const auto &d : m_documents)
        total += d.chunks;
    return total;
}

int KnowledgeManager::addDocument(const QString &path, const QString &title)
{
    Document d;
    d.id = m_nextId++;
    d.path = path;
    d.title = title.isEmpty() ? QFileInfo(path).fileName() : title;
    d.type = detectFileType(path);
    d.chunks = 10 + (path.length() % 20); /* Simulated chunk count */
    d.sizeBytes = 1024 + (path.length() * 100); /* Simulated size */
    d.indexed = QDateTime::currentDateTime();
    m_documents.append(d);
    emit documentsChanged();
    emit indexChanged();
    return d.id;
}

bool KnowledgeManager::removeDocument(int id)
{
    for (int i = 0; i < m_documents.size(); i++) {
        if (m_documents[i].id == id) {
            m_documents.removeAt(i);
            emit documentsChanged();
            emit indexChanged();
            return true;
        }
    }
    return false;
}

QVariantList KnowledgeManager::search(const QString &query, int maxResults)
{
    QVariantList results;
    QString q = query.toLower();

    for (const auto &d : m_documents) {
        if (d.title.toLower().contains(q) || d.path.toLower().contains(q) || d.type.toLower().contains(q)) {
            QVariantMap m;
            m["docId"] = d.id;
            m["title"] = d.title;
            m["path"] = d.path;
            m["type"] = d.type;
            m["relevance"] = d.title.toLower().contains(q) ? 0.95 : 0.70;
            m["snippet"] = "... " + d.title + " contains relevant information about " + query + " ...";
            results.append(m);
            if (results.size() >= maxResults) break;
        }
    }

    emit searchCompleted(results);
    return results;
}

QVariantList KnowledgeManager::getRelated(int docId, int maxResults)
{
    QVariantList results;
    QString targetType;
    for (const auto &d : m_documents) {
        if (d.id == docId) {
            targetType = d.type;
            break;
        }
    }

    for (const auto &d : m_documents) {
        if (d.id != docId && (d.type == targetType || results.size() < 2)) {
            QVariantMap m;
            m["docId"] = d.id;
            m["title"] = d.title;
            m["type"] = d.type;
            m["similarity"] = d.type == targetType ? 0.85 : 0.50;
            results.append(m);
            if (results.size() >= maxResults) break;
        }
    }
    return results;
}

void KnowledgeManager::reindex()
{
    m_indexStatus = "indexing";
    emit indexChanged();

    /* Simulate reindexing */
    for (auto &d : m_documents) {
        d.chunks = 10 + (d.path.length() % 20);
        d.indexed = QDateTime::currentDateTime();
    }

    m_indexStatus = "ready";
    emit indexChanged();
    emit documentsChanged();
}

QVariantMap KnowledgeManager::getDocumentDetail(int id)
{
    for (const auto &d : m_documents) {
        if (d.id == id) {
            QVariantMap m;
            m["id"] = d.id;
            m["path"] = d.path;
            m["title"] = d.title;
            m["type"] = d.type;
            m["chunks"] = d.chunks;
            m["sizeBytes"] = d.sizeBytes;
            m["indexed"] = d.indexed.toString("yyyy-MM-dd hh:mm:ss");
            return m;
        }
    }
    return {};
}

QString KnowledgeManager::askQuestion(const QString &question)
{
    /* Simulate RAG-based question answering */
    QVariantList results = search(question, 3);
    if (results.isEmpty()) {
        return "I couldn't find relevant information in the knowledge base for: " + question;
    }

    QString answer = "Based on the knowledge base:\n\n";
    for (const auto &r : results) {
        QVariantMap m = r.toMap();
        answer += "- From \"" + m["title"].toString() + "\": " + m["snippet"].toString() + "\n";
    }
    answer += "\nThis answer was generated using " + QString::number(results.size()) + " relevant document(s).";

    emit questionAnswered(answer);
    return answer;
}

QString KnowledgeManager::detectFileType(const QString &path) const
{
    QString ext = QFileInfo(path).suffix().toLower();
    if (ext == "md" || ext == "markdown") return "md";
    if (ext == "txt" || ext == "text") return "txt";
    if (ext == "pdf") return "pdf";
    if (ext == "c" || ext == "cpp" || ext == "h" || ext == "hpp") return "code";
    if (ext == "py" || ext == "js" || ext == "ts") return "code";
    if (ext == "qml") return "code";
    if (ext == "json" || ext == "yaml" || ext == "yml") return "config";
    return "other";
}

void KnowledgeManager::loadDefaults()
{
    addDocument("/docs/architecture_2025.md", "NeuralOS Architecture Guide");
    addDocument("/docs/api_reference.md", "NPIE API Reference");
    addDocument("/docs/getting_started.md", "Getting Started Guide");
    addDocument("/docs/security.md", "Security Documentation");
    addDocument("/docs/competitive_analysis.md", "Competitive OS Analysis");
    addDocument("/src/npie/api/npie.h", "NPIE C API Header");
    addDocument("/src/dashboard/qml/Theme.qml", "UI Theme Definition");
    addDocument("/README.md", "NeuralOS README");
}
