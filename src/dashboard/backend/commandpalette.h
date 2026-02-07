#ifndef COMMANDPALETTE_H
#define COMMANDPALETTE_H

#include <QObject>
#include <QVariantList>
#include <QStringList>

/**
 * Natural Language Command Palette (Bytebot OS-inspired)
 * OS-wide natural language control: open apps, change settings, search files
 */
class CommandPalette : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList suggestions READ suggestions NOTIFY suggestionsChanged)
    Q_PROPERTY(QStringList history READ history NOTIFY historyChanged)
    Q_PROPERTY(bool isActive READ isActive WRITE setIsActive NOTIFY activeChanged)

public:
    explicit CommandPalette(QObject *parent = nullptr);

    QVariantList suggestions() const;
    QStringList history() const;
    bool isActive() const;
    void setIsActive(bool v);

    Q_INVOKABLE QVariantMap execute(const QString &command);
    Q_INVOKABLE QVariantList getSuggestions(const QString &input);
    Q_INVOKABLE void clearHistory();

signals:
    void suggestionsChanged();
    void historyChanged();
    void activeChanged();
    void commandExecuted(const QString &action, const QString &target);

private:
    struct Command {
        QString pattern;
        QString action;
        QString description;
        QString icon;
    };

    QList<Command> m_commands;
    QStringList m_history;
    QVariantList m_suggestions;
    bool m_active = false;

    void registerCommands();
    bool fuzzyMatch(const QString &input, const QString &pattern) const;
};

#endif
