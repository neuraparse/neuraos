/**
 * NeuralOS Desktop Shell
 * Qt5 QML-based modern OS interface for edge AI devices
 */

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QFont>
#include <QDir>

#include "backend/systeminfo.h"
#include "backend/npiebridge.h"
#include "backend/npumonitor.h"
#include "backend/processmanager.h"
#include "backend/networkmanager.h"
#include "backend/settingsmanager.h"
#include "backend/aibusmanager.h"
#include "backend/aimemorymanager.h"
#include "backend/commandpalette.h"
#include "backend/automationmanager.h"
#include "backend/mcpmanager.h"
#include "backend/knowledgemanager.h"
#include "backend/ecosystemmanager.h"

int main(int argc, char *argv[])
{
    /* Auto-detect display platform: prefer wayland, fallback to xcb */
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        if (!qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY"))
            qputenv("QT_QPA_PLATFORM", "wayland");
        else if (!qEnvironmentVariableIsEmpty("DISPLAY"))
            qputenv("QT_QPA_PLATFORM", "xcb");
    }
    qputenv("QT_WAYLAND_DISABLE_WINDOWDECORATION", "1");
    qputenv("QT_QUICK_CONTROLS_STYLE", "Basic");

    QGuiApplication app(argc, argv);
    app.setApplicationName("NeuralOS");
    app.setApplicationVersion("2.0.0");
    app.setOrganizationName("NeuraParse");

    QFont defaultFont("Liberation Sans", 11);
    app.setFont(defaultFont);

    /* Create backend singletons */
    SystemInfo systemInfo;
    NPIEBridge npieBridge;
    NPUMonitor npuMonitor;
    ProcessManager processManager;
    NetworkManager networkManager;
    SettingsManager settingsManager;
    AIBusManager aiBusManager;
    AIMemoryManager aiMemoryManager;
    CommandPalette commandPalette;
    AutomationManager automationManager;
    MCPManager mcpManager;
    KnowledgeManager knowledgeManager;
    EcosystemManager ecosystemManager;

    /* QML engine */
    QQmlApplicationEngine engine;

    /* Expose backends to QML */
    engine.rootContext()->setContextProperty("SystemInfo", &systemInfo);
    engine.rootContext()->setContextProperty("NPIE", &npieBridge);
    engine.rootContext()->setContextProperty("NPUMonitor", &npuMonitor);
    engine.rootContext()->setContextProperty("ProcessManager", &processManager);
    engine.rootContext()->setContextProperty("NetworkManager", &networkManager);
    engine.rootContext()->setContextProperty("Settings", &settingsManager);
    engine.rootContext()->setContextProperty("AIBus", &aiBusManager);
    engine.rootContext()->setContextProperty("AIMemory", &aiMemoryManager);
    engine.rootContext()->setContextProperty("CommandEngine", &commandPalette);
    engine.rootContext()->setContextProperty("Automation", &automationManager);
    engine.rootContext()->setContextProperty("MCP", &mcpManager);
    engine.rootContext()->setContextProperty("Knowledge", &knowledgeManager);
    engine.rootContext()->setContextProperty("Ecosystem", &ecosystemManager);

    /* Add QML import paths so all files can find Theme singleton */
    engine.addImportPath(QStringLiteral("qrc:/qml"));
    engine.addImportPath(QStringLiteral("qrc:/"));

    /* Load QML */
    const QUrl url(QStringLiteral("qrc:/qml/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}
