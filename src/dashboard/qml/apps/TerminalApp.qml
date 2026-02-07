import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: termApp
    anchors.fill: parent

    property int activeTab: 0
    property var commandHistory: []
    property int historyIndex: -1

    ListModel { id: outputModel }

    /* ── Seed mock history on load ── */
    Component.onCompleted: {
        appendLine("NeuralOS v4.0.0 \u2014 AI-Native Terminal", Theme.primary, true)
        appendLine("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", Theme.textMuted, false)
        runCommand("neofetch")
        appendPrompt("ls")
        runBuiltin("ls")
        appendPrompt("npu-info")
        runBuiltin("npu-info")
        appendLine("", Theme.text, false)
    }

    function appendLine(text, color, bold) {
        var c = (typeof color === "string") ? color : (color ? color.toString() : "#EAEDF3")
        outputModel.append({ text: text, lineColor: c, bold: bold })
    }

    function appendPrompt(cmd) {
        outputModel.append({ text: "user@neuraos", color: Theme.success.toString(), bold: true })
        outputModel.append({ text: ":~$ " + cmd, color: Theme.primary.toString(), bold: false })
    }

    function runCommand(cmd) {
        appendPrompt(cmd)
        runBuiltin(cmd)
    }

    function runBuiltin(cmd) {
        var c = cmd.trim()
        if (c === "") return

        if (c === "clear") {
            outputModel.clear()
        } else if (c === "help") {
            appendLine("Available commands:", Theme.warning.toString(), true)
            appendLine("  help        Show this help message", Theme.text.toString(), false)
            appendLine("  clear       Clear terminal output", Theme.text.toString(), false)
            appendLine("  ls          List directory contents", Theme.text.toString(), false)
            appendLine("  pwd         Print working directory", Theme.text.toString(), false)
            appendLine("  whoami      Print current user", Theme.text.toString(), false)
            appendLine("  uname -a    Show system information", Theme.text.toString(), false)
            appendLine("  neofetch    Display system info banner", Theme.text.toString(), false)
            appendLine("  npu-info    Show NPU device status", Theme.text.toString(), false)
            appendLine("  ai-query    Ask the onboard AI a question", Theme.text.toString(), false)
        } else if (c === "ls") {
            appendLine("drwxr-xr-x  models/       drwxr-xr-x  configs/", "#5B9AFF", false)
            appendLine("drwxr-xr-x  agents/       drwxr-xr-x  logs/", "#5B9AFF", false)
            appendLine("-rw-r--r--  main.py       -rw-r--r--  setup.cfg", Theme.success.toString(), false)
            appendLine("-rwxr-xr-x  build.sh      -rw-r--r--  README.md", Theme.success.toString(), false)
            appendLine("-rw-r--r--  neural.conf   -rw-r--r--  Makefile", Theme.text.toString(), false)
        } else if (c === "pwd") {
            appendLine("/root/neuraos", Theme.text.toString(), false)
        } else if (c === "whoami") {
            appendLine("root", Theme.text.toString(), false)
        } else if (c === "uname -a" || c === "uname") {
            appendLine("NeuralOS 4.0.0 neuraos-desktop aarch64 NPU-Accelerated GNU/Linux", Theme.text.toString(), false)
        } else if (c === "neofetch") {
            appendLine("  _   _                      ___  ____  ", Theme.primary.toString(), true)
            appendLine(" | \\ | | ___ _   _ _ __ __ _/ _ \\/ ___| ", Theme.primary.toString(), true)
            appendLine(" |  \\| |/ _ \\ | | | '__/ _` | | | \\___ \\", Theme.primary.toString(), true)
            appendLine(" | |\\  |  __/ |_| | | | (_| | |_| |___) |", Theme.primary.toString(), true)
            appendLine(" |_| \\_|\\___|\\__,_|_|  \\__,_|\\___/|____/ ", Theme.primary.toString(), true)
            appendLine("", Theme.text.toString(), false)
            appendLine("  OS:      NeuralOS 4.0.0 Desktop", Theme.text.toString(), false)
            appendLine("  Kernel:  6.8.0-neural-aarch64", Theme.text.toString(), false)
            appendLine("  Host:    neuraos-desktop", Theme.textDim.toString(), false)
            appendLine("  Uptime:  3 hours, 42 min", Theme.textDim.toString(), false)
            appendLine("  CPU:     ARM Cortex-X4 @ 3.4 GHz (12%)", Theme.success.toString(), false)
            appendLine("  Memory:  4.2 / 16.0 GB (26%)", Theme.success.toString(), false)
            appendLine("  NPU:     Hexagon V79  \u2502  18 TOPS", Theme.secondary.toString(), false)
            appendLine("  NPIE:    v2.4.1 runtime active", Theme.secondary.toString(), false)
        } else if (c === "npu-info") {
            appendLine("\u250C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510", Theme.textMuted.toString(), false)
            appendLine("\u2502  NPU Device Status                    \u2502", Theme.warning.toString(), true)
            appendLine("\u251C\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524", Theme.textMuted.toString(), false)
            appendLine("\u2502  Device:   Hexagon V79 NPU             \u2502", Theme.text.toString(), false)
            appendLine("\u2502  Status:   ONLINE                      \u2502", Theme.success.toString(), true)
            appendLine("\u2502  Compute:  18 TOPS (INT8)              \u2502", Theme.text.toString(), false)
            appendLine("\u2502  Thermal:  42\u00B0C (nominal)              \u2502", Theme.success.toString(), false)
            appendLine("\u2502  Load:     34%  \u2588\u2588\u2588\u2588\u2591\u2591\u2591\u2591\u2591\u2591            \u2502", Theme.primary.toString(), false)
            appendLine("\u2502  Models:   3 loaded, 1 queued          \u2502", Theme.textDim.toString(), false)
            appendLine("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518", Theme.textMuted.toString(), false)
        } else if (c.indexOf("ai-query") === 0) {
            var query = c.substring(8).trim()
            if (query.length === 0) {
                appendLine("Usage: ai-query <your question>", Theme.warning.toString(), false)
            } else {
                appendLine("[AI] Processing query: \"" + query + "\"", Theme.secondary.toString(), true)
                appendLine("[AI] Response: The onboard NPU model suggests that " +
                    query.toLowerCase() + " can be addressed through the NeuralOS " +
                    "inference pipeline. Run `npu-info` for device status.", Theme.text.toString(), false)
            }
        } else {
            appendLine("sh: " + c + ": command not found. Type 'help' for available commands.", Theme.error.toString(), false)
        }
        appendLine("", Theme.text.toString(), false)
    }

    /* ── Glass background ── */
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.92)
        radius: Theme.radiusSmall

        /* ── Tab bar ── */
        Rectangle {
            id: tabBar
            anchors { top: parent.top; left: parent.left; right: parent.right }
            height: 38
            color: Theme.surfaceAlt
            radius: Theme.radiusSmall

            Rectangle {
                anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                height: parent.radius
                color: parent.color
            }

            Row {
                id: tabRow
                anchors { left: parent.left; leftMargin: 8; verticalCenter: parent.verticalCenter }
                spacing: 2

                Repeater {
                    model: [
                        { label: "~", icon: "home" },
                        { label: "build/", icon: "code" },
                        { label: "logs/", icon: "file" }
                    ]
                    delegate: Rectangle {
                        width: tabLabel.implicitWidth + 48
                        height: 30
                        radius: Theme.radiusTiny
                        color: termApp.activeTab === index ? Theme.surfaceLight : "transparent"

                        Row {
                            anchors.centerIn: parent
                            spacing: 5
                            Components.CanvasIcon {
                                iconName: modelData.icon
                                iconSize: 13
                                iconColor: termApp.activeTab === index ? Theme.primary : Theme.textMuted
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Text {
                                id: tabLabel
                                text: modelData.label
                                font.family: "monospace"
                                font.pixelSize: 12
                                color: termApp.activeTab === index ? Theme.text : Theme.textDim
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        /* Active underline */
                        Rectangle {
                            anchors { bottom: parent.bottom; horizontalCenter: parent.horizontalCenter }
                            width: parent.width - 16
                            height: 2
                            radius: 1
                            color: Theme.primary
                            visible: termApp.activeTab === index
                        }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: termApp.activeTab = index
                        }

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }
                    }
                }
            }

            /* New-tab button */
            Rectangle {
                anchors {
                    left: tabRow.right; leftMargin: 6
                    verticalCenter: parent.verticalCenter
                }
                width: 24; height: 24
                radius: Theme.radiusTiny
                color: plusMa.containsMouse ? Theme.surfaceLight : "transparent"
                Components.CanvasIcon {
                    anchors.centerIn: parent
                    iconName: "plus"; iconSize: 13; iconColor: Theme.textMuted
                }
                MouseArea {
                    id: plusMa; anchors.fill: parent
                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                }
            }
        }

        /* ── Terminal output area ── */
        Rectangle {
            id: termArea
            anchors {
                top: tabBar.bottom; left: parent.left; right: parent.right
                bottom: inputBar.top; margins: 0
            }
            color: Qt.rgba(Theme.background.r, Theme.background.g, Theme.background.b, 0.95)

            Flickable {
                id: termFlick
                anchors.fill: parent
                anchors.margins: 10
                contentHeight: outputCol.height
                clip: true
                boundsBehavior: Flickable.StopAtBounds

                Column {
                    id: outputCol
                    width: termFlick.width
                    spacing: 0

                    Repeater {
                        model: outputModel
                        delegate: Text {
                            width: outputCol.width
                            text: model.text
                            color: model.lineColor || "#EAEDF3"
                            font.family: "monospace"
                            font.pixelSize: 13
                            font.bold: model.bold
                            wrapMode: Text.WrapAnywhere
                            textFormat: Text.PlainText
                            height: implicitHeight > 0 ? implicitHeight : 4
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AsNeeded
                    minimumSize: 0.06
                    contentItem: Rectangle {
                        implicitWidth: 5
                        radius: 3
                        color: Theme.textMuted
                        opacity: 0.5
                    }
                }

                onContentHeightChanged: {
                    if (contentHeight > height)
                        contentY = contentHeight - height
                }
            }
        }

        /* ── Input bar ── */
        Rectangle {
            id: inputBar
            anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
            height: 40
            color: Theme.surfaceAlt
            radius: Theme.radiusSmall

            Rectangle {
                anchors { left: parent.left; right: parent.right; top: parent.top }
                height: parent.radius
                color: parent.color
            }

            Rectangle {
                anchors { left: parent.left; right: parent.right; top: parent.top }
                height: 1
                color: Theme.surfaceLight
            }

            Row {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 0

                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: "user@neuraos"
                    color: Theme.success
                    font.family: "monospace"
                    font.pixelSize: 13
                    font.bold: true
                }
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: ":~$ "
                    color: Theme.primary
                    font.family: "monospace"
                    font.pixelSize: 13
                    font.bold: true
                }
                TextInput {
                    id: cmdInput
                    anchors.verticalCenter: parent.verticalCenter
                    width: parent.width - 160
                    color: Theme.text
                    font.family: "monospace"
                    font.pixelSize: 13
                    clip: true
                    focus: true
                    selectByMouse: true
                    selectionColor: Theme.primary
                    selectedTextColor: "#FFFFFF"
                    cursorVisible: focus

                    Rectangle {
                        x: cmdInput.cursorRectangle ? cmdInput.cursorRectangle.x : 0
                        y: cmdInput.cursorRectangle ? cmdInput.cursorRectangle.y : 0
                        width: 2
                        height: cmdInput.cursorRectangle ? cmdInput.cursorRectangle.height : 16
                        color: Theme.primary
                        visible: cmdInput.activeFocus
                        SequentialAnimation on opacity {
                            running: true; loops: Animation.Infinite
                            NumberAnimation { to: 0; duration: 530 }
                            NumberAnimation { to: 1; duration: 530 }
                        }
                    }

                    Keys.onUpPressed: {
                        if (commandHistory.length > 0) {
                            if (historyIndex < commandHistory.length - 1) historyIndex++
                            text = commandHistory[commandHistory.length - 1 - historyIndex]
                        }
                    }
                    Keys.onDownPressed: {
                        if (historyIndex > 0) {
                            historyIndex--
                            text = commandHistory[commandHistory.length - 1 - historyIndex]
                        } else {
                            historyIndex = -1
                            text = ""
                        }
                    }

                    onAccepted: {
                        var cmd = text.trim()
                        if (cmd === "") return
                        commandHistory.push(cmd)
                        historyIndex = -1
                        appendPrompt(cmd)
                        runBuiltin(cmd)
                        text = ""
                        termFlick.contentY = Math.max(0, termFlick.contentHeight - termFlick.height)
                    }
                }
            }
        }
    }
}
