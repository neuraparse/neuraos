#include "commandpalette.h"

CommandPalette::CommandPalette(QObject *parent) : QObject(parent)
{
    registerCommands();
}

QVariantList CommandPalette::suggestions() const { return m_suggestions; }
QStringList CommandPalette::history() const { return m_history; }
bool CommandPalette::isActive() const { return m_active; }

void CommandPalette::setIsActive(bool v)
{
    if (m_active != v) {
        m_active = v;
        emit activeChanged();
    }
}

QVariantMap CommandPalette::execute(const QString &command)
{
    QVariantMap result;
    QString cmd = command.trimmed().toLower();

    /* Add to history */
    m_history.prepend(command);
    if (m_history.size() > 20)
        m_history = m_history.mid(0, 20);
    emit historyChanged();

    /* Match commands */
    for (const auto &c : m_commands) {
        if (fuzzyMatch(cmd, c.pattern)) {
            result["action"] = c.action;
            result["matched"] = true;
            result["description"] = c.description;

            /* Extract target from command */
            QString target;
            if (cmd.startsWith("open "))
                target = cmd.mid(5).trimmed();
            else if (cmd.startsWith("search "))
                target = cmd.mid(7).trimmed();
            else if (cmd.startsWith("set "))
                target = cmd.mid(4).trimmed();
            else if (cmd.startsWith("run "))
                target = cmd.mid(4).trimmed();
            result["target"] = target;

            emit commandExecuted(c.action, target);
            return result;
        }
    }

    result["matched"] = false;
    result["action"] = "unknown";
    result["description"] = "Command not recognized: " + command;
    return result;
}

QVariantList CommandPalette::getSuggestions(const QString &input)
{
    QVariantList results;
    if (input.isEmpty()) {
        /* Show recent history */
        for (int i = 0; i < qMin(5, m_history.size()); i++) {
            QVariantMap m;
            m["text"] = m_history[i];
            m["icon"] = "clock";
            m["type"] = "history";
            results.append(m);
        }
        m_suggestions = results;
        emit suggestionsChanged();
        return results;
    }

    QString q = input.toLower();
    for (const auto &c : m_commands) {
        if (fuzzyMatch(q, c.pattern) || c.description.toLower().contains(q)) {
            QVariantMap m;
            m["text"] = c.pattern;
            m["icon"] = c.icon;
            m["type"] = "command";
            m["description"] = c.description;
            m["action"] = c.action;
            results.append(m);
        }
    }

    m_suggestions = results;
    emit suggestionsChanged();
    return results;
}

void CommandPalette::clearHistory()
{
    m_history.clear();
    emit historyChanged();
}

void CommandPalette::registerCommands()
{
    m_commands = {
        /* App launchers */
        {"open terminal",       "open_app", "Open Terminal application",        "terminal"},
        {"open file manager",   "open_app", "Open File Manager",               "folder"},
        {"open settings",       "open_app", "Open Settings",                   "gear"},
        {"open system monitor", "open_app", "Open System Monitor",             "monitor"},
        {"open neural studio",  "open_app", "Open Neural Studio AI IDE",       "neural"},
        {"open ai assistant",   "open_app", "Open AI Assistant chat",          "robot"},
        {"open ai bus",         "open_app", "Open AI Bus orchestration",       "hub"},
        {"open ai memory",      "open_app", "Open AI Memory browser",          "brain"},
        {"open automation",     "open_app", "Open Automation Studio",          "automation"},
        {"open mcp hub",        "open_app", "Open MCP Integration Hub",        "plug"},
        {"open knowledge base", "open_app", "Open Knowledge Base & RAG",       "book"},
        {"open ecosystem",      "open_app", "Open Ecosystem Manager",          "network"},
        {"open calculator",     "open_app", "Open Calculator",                 "grid"},
        {"open browser",        "open_app", "Open Web Browser",                "globe"},
        {"open notes",          "open_app", "Open Notes",                      "edit"},
        {"open music",          "open_app", "Open Music Player",               "volume"},

        /* System commands */
        {"system status",       "system",   "Show system CPU, RAM, disk status",  "monitor"},
        {"npu status",          "system",   "Show NPU utilization and models",    "chip"},
        {"network status",      "system",   "Show network connectivity",          "wifi"},

        /* Settings */
        {"set brightness",      "setting",  "Adjust display brightness",       "monitor"},
        {"set volume",          "setting",  "Adjust audio volume",             "volume"},
        {"set theme dark",      "setting",  "Switch to dark theme",            "monitor"},
        {"set theme light",     "setting",  "Switch to light theme",           "monitor"},

        /* AI commands */
        {"run inference",       "ai",       "Run AI model inference",          "neural"},
        {"search files",        "search",   "Search for files on the system",  "folder"},
        {"search knowledge",    "search",   "Search knowledge base",           "book"},

        /* Workflow */
        {"record workflow",     "automation","Start recording a new workflow",  "automation"},
        {"stop recording",      "automation","Stop workflow recording",         "automation"},
    };
}

bool CommandPalette::fuzzyMatch(const QString &input, const QString &pattern) const
{
    if (pattern.contains(input)) return true;
    if (input.contains(pattern)) return true;

    /* Simple word-based matching */
    QStringList inputWords = input.split(' ', Qt::SkipEmptyParts);
    QStringList patternWords = pattern.split(' ', Qt::SkipEmptyParts);

    int matched = 0;
    for (const auto &iw : inputWords) {
        for (const auto &pw : patternWords) {
            if (pw.startsWith(iw) || iw.startsWith(pw)) {
                matched++;
                break;
            }
        }
    }
    return matched >= qMax(1, inputWords.size() / 2);
}
