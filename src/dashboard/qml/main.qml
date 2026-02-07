import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "."
import "shell" as Shell
import "shell/WindowManager.js" as WM

ApplicationWindow {
    id: root
    visible: true
    width: 1280; height: 720
    minimumWidth: 800; minimumHeight: 480
    title: "NeuralOS"
    color: Theme.background

    font.family: Theme.fontFamily

    Behavior on color { ColorAnimation { duration: Theme.animNormal } }

    property bool startMenuOpen: false
    property bool notifCenterOpen: false
    property int windowUpdateTick: 0
    property var recentApps: []

    /* ─── App Registry ─── */
    property var appRegistry: [
        { source: "SystemMonitorApp.qml", title: "System Monitor", icon: "monitor", color: "#F59E0B", category: "System", width: 850, height: 550, singleton: true },
        { source: "TerminalApp.qml", title: "Terminal", icon: "terminal", color: "#10B981", category: "System", width: 800, height: 500, singleton: false },
        { source: "FileManagerApp.qml", title: "File Manager", icon: "folder", color: "#3B82F6", category: "System", width: 850, height: 550, singleton: true },
        { source: "SettingsApp.qml", title: "Settings", icon: "gear", color: "#7C3AED", category: "System", width: 800, height: 550, singleton: true },
        { source: "NeuralStudio.qml", title: "Neural Studio", icon: "neural", color: "#00D9FF", category: "AI & ML", width: 900, height: 600, singleton: true },
        { source: "AIAgentHub.qml", title: "AI Agent Hub", icon: "robot", color: "#EC4899", category: "AI & ML", width: 900, height: 600, singleton: true },
        { source: "NPUControlCenter.qml", title: "NPU Control", icon: "chip", color: "#F59E0B", category: "AI & ML", width: 800, height: 550, singleton: true },
        { source: "DroneCommandCenter.qml", title: "Drone Command", icon: "drone", color: "#06B6D4", category: "Defense", width: 950, height: 650, singleton: true },
        { source: "DefenseMonitor.qml", title: "Defense Monitor", icon: "shield", color: "#EF4444", category: "Defense", width: 900, height: 600, singleton: true },
        { source: "QuantumLab.qml", title: "Quantum Lab", icon: "atom", color: "#A78BFA", category: "Quantum", width: 900, height: 600, singleton: true },
        { source: "NetworkCenter.qml", title: "Network Center", icon: "wifi", color: "#06B6D4", category: "System", width: 750, height: 500, singleton: true },
        { source: "PackageManager.qml", title: "Package Manager", icon: "box", color: "#F97316", category: "System", width: 800, height: 550, singleton: true },
        { source: "AppStore.qml", title: "App Store", icon: "apps", color: "#4C8FFF", category: "System", width: 900, height: 600, singleton: true },
        { source: "CalculatorApp.qml", title: "Calculator", icon: "grid", color: "#F59E0B", category: "Utilities", width: 340, height: 520, singleton: true },
        { source: "TextEditorApp.qml", title: "Text Editor", icon: "edit", color: "#10B981", category: "Utilities", width: 800, height: 550, singleton: false },
        { source: "WebBrowserApp.qml", title: "Web Browser", icon: "globe", color: "#3B82F6", category: "Internet", width: 1000, height: 650, singleton: false },
        { source: "ImageViewerApp.qml", title: "Image Viewer", icon: "image", color: "#A855F7", category: "Utilities", width: 800, height: 550, singleton: false },
        { source: "MusicPlayerApp.qml", title: "Music Player", icon: "volume", color: "#EC4899", category: "Media", width: 750, height: 450, singleton: true },
        { source: "TaskManagerApp.qml", title: "Task Manager", icon: "dashboard", color: "#EF4444", category: "System", width: 900, height: 600, singleton: true },
        { source: "NotesApp.qml", title: "Notes", icon: "edit", color: "#10B981", category: "Utilities", width: 800, height: 550, singleton: true },
        { source: "CalendarApp.qml", title: "Calendar", icon: "calendar", color: "#3B82F6", category: "Utilities", width: 750, height: 550, singleton: true },
        { source: "AIAssistantApp.qml", title: "AI Assistant", icon: "robot", color: "#5B9AFF", category: "AI & ML", width: 900, height: 600, singleton: true },
        { source: "WeatherApp.qml", title: "Weather", icon: "weather", color: "#38BDF8", category: "Utilities", width: 650, height: 500, singleton: true },
        { source: "PhotosApp.qml", title: "Photos", icon: "photo", color: "#F472B6", category: "Media", width: 850, height: 600, singleton: true },
        { source: "ClockApp.qml", title: "Clock", icon: "clock", color: "#A78BFA", category: "Utilities", width: 600, height: 500, singleton: true },
        { source: "VideoPlayerApp.qml", title: "Video Player", icon: "video", color: "#EF4444", category: "Media", width: 900, height: 600, singleton: true }
    ]

    /* ─── Open App Function ─── */
    function openApp(appDef) {
        startMenuOpen = false

        /* Track recent apps */
        var found = -1
        for (var r = 0; r < recentApps.length; r++) {
            if (recentApps[r].source === appDef.source) { found = r; break }
        }
        var newRecent = recentApps.slice()
        if (found >= 0) newRecent.splice(found, 1)
        newRecent.unshift(appDef)
        if (newRecent.length > 5) newRecent = newRecent.slice(0, 5)
        recentApps = newRecent

        var wid = WM.openWindow(appDef)
        var winData = WM.getWindow(wid)

        var framePath = "qrc:/qml/shell/WindowFrame.qml"
        var appPath = "qrc:/qml/apps/" + appDef.source

        var frameComp = Qt.createComponent(framePath)
        var appComp = Qt.createComponent(appPath)

        if (frameComp.status !== Component.Ready) {
            console.log("WindowFrame error:", frameComp.errorString())
            return
        }
        if (appComp.status !== Component.Ready) {
            console.log("App error (" + appDef.source + "):", appComp.errorString())
            return
        }

        var maxW = windowLayer.width - 40
        var maxH = windowLayer.height - 40
        var fw = Math.min(appDef.width || 800, maxW)
        var fh = Math.min(appDef.height || 500, maxH)

        var frame = frameComp.createObject(windowLayer, {
            windowId: wid,
            windowTitle: appDef.title,
            windowIcon: appDef.icon,
            windowAccent: appDef.color,
            x: winData.x,
            y: winData.y,
            width: fw,
            height: fh,
            z: winData.zOrder,
            isFocused: true,
            windowLayer: windowLayer
        })

        var app = appComp.createObject(frame.contentItem)

        winData.item = frame

        /* Close with animation */
        frame.closeRequested.connect(function() {
            frame.animateClose()
        })
        frame.closeAnimFinished.connect(function() {
            WM.closeWindow(wid)
            windowUpdateTick++
        })

        /* Minimize with animation */
        frame.minimizeRequested.connect(function() {
            frame.animateMinimize()
            WM.minimizeWindow(wid)
            windowUpdateTick++
        })

        frame.maximizeRequested.connect(function() {
            var isMax = WM.toggleMaximize(wid)
            if (isMax) {
                frame.savedX = frame.x; frame.savedY = frame.y
                frame.savedW = frame.width; frame.savedH = frame.height
                frame.x = 0; frame.y = 0
                frame.width = windowLayer.width
                frame.height = windowLayer.height
                frame.isMaximized = true
            } else {
                frame.x = frame.savedX; frame.y = frame.savedY
                frame.width = frame.savedW; frame.height = frame.savedH
                frame.isMaximized = false
            }
        })

        frame.focusRequested.connect(function() {
            var newZ = WM.focusWindow(wid)
            frame.z = newZ
            frame.isFocused = true
            for (var i = 0; i < windowLayer.children.length; i++) {
                var child = windowLayer.children[i]
                if (child !== frame && child.windowId !== undefined) {
                    child.isFocused = false
                }
            }
            windowUpdateTick++
        })

        windowUpdateTick++
    }

    function handleTaskbarWindowClick(winId) {
        var winData = WM.getWindow(winId)
        if (!winData) return
        if (winData.minimized) {
            WM.restoreWindow(winId)
            if (winData.item && winData.item.animateRestore) {
                winData.item.animateRestore()
            }
        } else {
            WM.focusWindow(winId)
            if (winData.item) {
                winData.item.z = winData.zOrder
                winData.item.isFocused = true
            }
        }
        for (var i = 0; i < windowLayer.children.length; i++) {
            var child = windowLayer.children[i]
            if (child.windowId !== undefined && child.windowId !== winId) {
                child.isFocused = false
            }
        }
        windowUpdateTick++
    }

    function openAppBySource(source) {
        for (var i = 0; i < appRegistry.length; i++) {
            if (appRegistry[i].source === source) {
                openApp(appRegistry[i])
                return
            }
        }
    }

    /* ─── Desktop Surface ─── */
    Shell.Desktop {
        id: desktop
        anchors.fill: parent
        z: 1
    }

    /* ─── Desktop Widgets ─── */
    Item {
        id: widgetLayer
        anchors.fill: parent
        anchors.bottomMargin: Theme.taskbarH + 16
        z: 10

        Loader {
            id: clockWidget
            x: parent.width - 220; y: 16
            source: "qrc:/qml/shell/widgets/ClockWidget.qml"
        }

        Loader {
            id: sysWidget
            x: parent.width - 240; y: 130
            source: "qrc:/qml/shell/widgets/SystemStatsWidget.qml"
        }

        Loader {
            id: aiWidget
            x: parent.width - 220; y: 310
            source: "qrc:/qml/shell/widgets/AIStatusWidget.qml"
        }

        Loader {
            id: quickWidget
            x: 16; y: 16
            source: "qrc:/qml/shell/widgets/QuickActionsWidget.qml"
            onLoaded: {
                if (item && item.launchApp) {
                    item.launchApp.connect(function(src) { root.openAppBySource(src) })
                }
            }
        }

        Loader {
            id: weatherWidget
            x: 16; y: 170
            source: "qrc:/qml/shell/widgets/WeatherWidget.qml"
        }

        Loader {
            id: mediaWidget
            x: 16; y: 310
            source: "qrc:/qml/shell/widgets/MediaWidget.qml"
        }

        Loader {
            id: calendarWidget
            x: parent.width - 230; y: 460
            source: "qrc:/qml/shell/widgets/CalendarWidget.qml"
        }
    }

    /* ─── Window Layer ─── */
    Item {
        id: windowLayer
        anchors.fill: parent
        anchors.bottomMargin: Theme.taskbarH + 16
        z: 50
    }

    /* ─── Overlay Backdrop ─── */
    MouseArea {
        anchors.fill: parent
        visible: startMenuOpen || notifCenterOpen
        z: 199
        onClicked: {
            startMenuOpen = false
            notifCenterOpen = false
        }
    }

    /* ─── Start Menu (animated) ─── */
    Shell.StartMenu {
        id: startMenu
        x: taskbar.x
        y: root.startMenuOpen ? (root.height - Theme.taskbarH - height - 16) : (root.height - Theme.taskbarH - height - 6)
        opacity: root.startMenuOpen ? 1.0 : 0.0
        scale: root.startMenuOpen ? 1.0 : 0.95
        visible: opacity > 0
        z: 200

        Behavior on y { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }
        Behavior on opacity { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }
        Behavior on scale { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }

        appList: root.appRegistry
        pinnedApps: root.appRegistry.slice(0, 6)
        recentApps: root.recentApps
        onAppClicked: root.openApp(appDef)
        onCloseMenu: root.startMenuOpen = false
    }

    /* ─── Notification Center (animated slide) ─── */
    Shell.NotificationCenter {
        id: notifCenter
        x: root.notifCenterOpen ? (root.width - width - 4) : (root.width + 10)
        y: 4
        width: Theme.notifCenterW
        height: root.height - Theme.taskbarH - 24
        opacity: root.notifCenterOpen ? 1.0 : 0.0
        visible: opacity > 0
        z: 200

        Behavior on x { NumberAnimation { duration: 250; easing.type: Easing.OutCubic } }
        Behavior on opacity { NumberAnimation { duration: 250; easing.type: Easing.OutCubic } }

        onClosePanel: root.notifCenterOpen = false
    }

    /* ─── Context Menu ─── */
    Shell.ContextMenu {
        id: contextMenu
        z: 250

        onItemClicked: {
            if (action === "terminal") openAppBySource("TerminalApp.qml")
            else if (action === "monitor") openAppBySource("SystemMonitorApp.qml")
            else if (action === "settings") openAppBySource("SettingsApp.qml")
            else if (action === "taskmanager") openAppBySource("TaskManagerApp.qml")
            else if (action === "filemanager") openAppBySource("FileManagerApp.qml")
            else if (action === "aiassistant") openAppBySource("AIAssistantApp.qml")
            else if (action === "wallpaper") openAppBySource("SettingsApp.qml")
            else if (action === "display") openAppBySource("SettingsApp.qml")
            else if (action.indexOf("tb_") === 0) {
                var parts = action.split("_")
                var tbAction = parts[1]
                var tbWinId = parseInt(parts[2])
                var tbWin = WM.getWindow(tbWinId)
                if (!tbWin || !tbWin.item) return
                if (tbAction === "close") tbWin.item.closeRequested()
                else if (tbAction === "minimize") tbWin.item.minimizeRequested()
                else if (tbAction === "maximize") tbWin.item.maximizeRequested()
                else if (tbAction === "restore") handleTaskbarWindowClick(tbWinId)
            }
        }
    }

    /* ─── Taskbar (Floating Dock) ─── */
    Shell.TaskBar {
        id: taskbar
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.72
        height: Theme.taskbarH
        z: 100

        startMenuOpen: root.startMenuOpen
        runningWindows: {
            windowUpdateTick;
            return WM.getWindowList()
        }

        onStartToggled: root.startMenuOpen = !root.startMenuOpen
        onNotificationToggled: root.notifCenterOpen = !root.notifCenterOpen
        onWindowClicked: root.handleTaskbarWindowClick(windowId)
        onWindowRightClicked: {
            contextMenu.show(globalX, globalY - 160, [
                { label: "Restore", icon: "fullscreen", action: "tb_restore_" + windowId },
                { label: "Minimize", icon: "minus", action: "tb_minimize_" + windowId },
                { label: "Maximize", icon: "fullscreen", action: "tb_maximize_" + windowId },
                { separator: true },
                { label: "Close", icon: "close", action: "tb_close_" + windowId, color: "#E84855" }
            ])
        }
    }

    /* ─── Desktop Right-click ─── */
    MouseArea {
        anchors.fill: parent
        anchors.bottomMargin: Theme.taskbarH + 16
        z: 0
        acceptedButtons: Qt.RightButton
        onClicked: {
            contextMenu.show(mouse.x, mouse.y, [
                { label: "Open Terminal", icon: "terminal", action: "terminal" },
                { label: "File Manager", icon: "folder", action: "filemanager" },
                { label: "AI Assistant", icon: "robot", action: "aiassistant" },
                { separator: true },
                { label: "System Monitor", icon: "monitor", action: "monitor" },
                { label: "Task Manager", icon: "dashboard", action: "taskmanager" },
                { separator: true },
                { label: "Change Wallpaper", icon: "image", action: "wallpaper" },
                { label: "Display Settings", icon: "monitor", action: "display" },
                { separator: true },
                { label: "Settings", icon: "gear", action: "settings" },
                { separator: true },
                { label: "About NeuralOS", icon: "info", action: "about", color: "#666" }
            ])
        }
    }

    /* ─── Keyboard Shortcuts ─── */
    Shortcut {
        sequence: "Ctrl+Alt+T"
        onActivated: openAppBySource("TerminalApp.qml")
    }

    Shortcut {
        sequence: "Escape"
        onActivated: {
            startMenuOpen = false
            notifCenterOpen = false
            contextMenu.hide()
        }
    }

    Shortcut {
        sequence: "Alt+F4"
        onActivated: {
            var fid = WM.getFocusedId()
            if (fid >= 0) {
                var fwin = WM.getWindow(fid)
                if (fwin && fwin.item) fwin.item.closeRequested()
            }
        }
    }

    Shortcut {
        sequence: "Meta+Space"
        onActivated: startMenuOpen = !startMenuOpen
    }

    Shortcut {
        sequence: "Meta+A"
        onActivated: openAppBySource("AIAssistantApp.qml")
    }

    Shortcut {
        sequence: "Meta+E"
        onActivated: openAppBySource("FileManagerApp.qml")
    }

    Shortcut {
        sequence: "Meta+T"
        onActivated: openAppBySource("TerminalApp.qml")
    }

    Shortcut {
        sequence: "Ctrl+Shift+Escape"
        onActivated: openAppBySource("TaskManagerApp.qml")
    }

    /* Global 1-second heartbeat */
    Timer {
        id: heartbeat
        interval: 1000; running: true; repeat: true
        onTriggered: windowUpdateTick++
    }
}
