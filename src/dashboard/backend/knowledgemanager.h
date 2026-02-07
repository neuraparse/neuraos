#ifndef KNOWLEDGEMANAGER_H
#define KNOWLEDGEMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QDateTime>

/**
 * Knowledge Base & RAG Manager (Archon OS + Kuse AI OS-inspired)
 * Local document indexing with AI-powered search
 */
class KnowledgeManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int totalDocs READ totalDocs NOTIFY documentsChanged)
    Q_PROPERTY(QString indexStatus READ indexStatus NOTIFY indexChanged)
    Q_PROPERTY(QVariantList documents READ documents NOTIFY documentsChanged)
    Q_PROPERTY(int totalChunks READ totalChunks NOTIFY indexChanged)

public:
    explicit KnowledgeManager(QObject *parent = nullptr);

    int totalDocs() const;
    QString indexStatus() const;
    QVariantList documents() const;
    int totalChunks() const;

    Q_INVOKABLE int addDocument(const QString &path, const QString &title);
    Q_INVOKABLE bool removeDocument(int id);
    Q_INVOKABLE QVariantList search(const QString &query, int maxResults = 10);
    Q_INVOKABLE QVariantList getRelated(int docId, int maxResults = 5);
    Q_INVOKABLE void reindex();
    Q_INVOKABLE QVariantMap getDocumentDetail(int id);
    Q_INVOKABLE QString askQuestion(const QString &question);

signals:
    void documentsChanged();
    void indexChanged();
    void searchCompleted(const QVariantList &results);
    void questionAnswered(const QString &answer);

private:
    struct Document {
        int id;
        QString path;
        QString title;
        QString type; // "txt", "md", "pdf", "code"
        int chunks;
        int sizeBytes;
        QDateTime indexed;
    };

    QList<Document> m_documents;
    QString m_indexStatus = "ready";
    int m_nextId = 1;

    void loadDefaults();
    QString detectFileType(const QString &path) const;
};

#endif
