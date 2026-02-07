#ifndef AIMEMORYMANAGER_H
#define AIMEMORYMANAGER_H

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QDateTime>

/**
 * Shared AI Memory Manager (Steve OS-inspired)
 * Cross-application contextual AI memory system
 */
class AIMemoryManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(int memoryEntries READ memoryEntries NOTIFY memoryChanged)
    Q_PROPERTY(int totalSize READ totalSize NOTIFY memoryChanged)
    Q_PROPERTY(QStringList categories READ categories NOTIFY memoryChanged)
    Q_PROPERTY(QVariantList entries READ entries NOTIFY memoryChanged)

public:
    explicit AIMemoryManager(QObject *parent = nullptr);

    int memoryEntries() const;
    int totalSize() const;
    QStringList categories() const;
    QVariantList entries() const;

    Q_INVOKABLE bool store(const QString &key, const QString &value, const QString &category, const QString &source);
    Q_INVOKABLE QString recall(const QString &key);
    Q_INVOKABLE QVariantList search(const QString &query);
    Q_INVOKABLE QVariantList getByCategory(const QString &category);
    Q_INVOKABLE QVariantMap getContext();
    Q_INVOKABLE bool removeEntry(const QString &key);
    Q_INVOKABLE void clearCategory(const QString &category);
    Q_INVOKABLE void clearAll();

signals:
    void memoryChanged();

private:
    struct MemoryEntry {
        QString key;
        QString value;
        QString category;
        QString source;
        QDateTime timestamp;
        int accessCount;
    };

    QList<MemoryEntry> m_entries;

    void loadDefaults();
};

#endif
