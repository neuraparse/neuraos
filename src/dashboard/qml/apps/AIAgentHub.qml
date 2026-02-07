import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: agentApp
    anchors.fill: parent

    property int selectedAgent: -1
    property var agentList: []

    function refreshAgents() {
        var procs = ProcessManager.processes
        var agents = []
        for (var i = 0; i < procs.length; i++) {
            var p = procs[i]
            var stateLabel = "Idle"
            if (p.state === "S" || p.state === "R") stateLabel = "Running"
            else if (p.state === "T") stateLabel = "Paused"
            else if (p.state === "Z") stateLabel = "Zombie"

            agents.push({
                name: p.name,
                atype: "PID " + p.pid,
                status: stateLabel,
                memory: (p.rss / 1024).toFixed(1) + " MB",
                pid: p.pid,
                rssKb: p.rss
            })
        }
        agents.sort(function(a, b) { return b.rssKb - a.rssKb })
        agentList = agents.slice(0, 25)
        agentListChanged()
    }

    Component.onCompleted: refreshAgents()

    Timer {
        interval: 3000; running: true; repeat: true
        onTriggered: refreshAgents()
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Top bar */
            Rectangle {
                Layout.fillWidth: true
                height: 46
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 10; spacing: 12

                    Text { text: "System Processes"; color: Theme.text; font.pixelSize: 14; font.bold: true }

                    Item { Layout.fillWidth: true }

                    /* Stats */
                    Row {
                        spacing: 16
                        StatPill { label: "Processes"; value: ProcessManager.processCount.toString(); pillColor: Theme.success }
                        StatPill { label: "Inferences"; value: NPIE.inferenceCount.toString(); pillColor: Theme.primary }
                        StatPill {
                            label: "Memory"
                            value: (SystemInfo.memoryUsed / 1048576).toFixed(0) + " MB"
                            pillColor: Theme.warning
                        }
                    }

                    /* Refresh button */
                    Rectangle {
                        width: 80; height: 30; radius: Theme.radiusSmall
                        color: refMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary
                        Text { anchors.centerIn: parent; text: "Refresh"; color: "#000"; font.pixelSize: 10; font.bold: true }
                        MouseArea { id: refMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { ProcessManager.refresh(); refreshAgents() } }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Agent list + detail */
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* Agent list */
                Flickable {
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width * 0.55
                    contentHeight: agentListCol.height
                    clip: true

                    Column {
                        id: agentListCol
                        width: parent.width
                        spacing: 1

                        Repeater {
                            model: agentList.length

                            Rectangle {
                                width: parent.width
                                height: 64
                                color: selectedAgent === index ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.08) :
                                       agMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                property var item: agentList[index]

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 10

                                    Rectangle {
                                        width: 40; height: 40; radius: 20
                                        color: item && item.status === "Running" ? Qt.rgba(0.06, 0.72, 0.51, 0.15) :
                                               item && item.status === "Paused" ? Qt.rgba(0.96, 0.62, 0.04, 0.15) :
                                               Qt.rgba(0.5, 0.5, 0.5, 0.15)

                                        Text {
                                            anchors.centerIn: parent
                                            text: "\u2699"
                                            font.pixelSize: 18
                                            color: item && item.status === "Running" ? Theme.success :
                                                   item && item.status === "Paused" ? Theme.warning : Theme.textDim
                                        }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: item ? item.name : ""; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                        Text { text: item ? item.atype : ""; color: Theme.textDim; font.pixelSize: 10 }
                                    }

                                    Components.StatusBadge {
                                        text: item ? item.status : ""
                                        badgeColor: item && item.status === "Running" ? Theme.success :
                                                    item && item.status === "Paused" ? Theme.warning : Theme.textDim
                                    }

                                    Text { text: item ? item.memory : ""; color: Theme.textDim; font.pixelSize: 10 }
                                }

                                MouseArea {
                                    id: agMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedAgent = index
                                }
                            }
                        }
                    }
                }

                Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

                /* Detail panel */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        visible: selectedAgent >= 0 && selectedAgent < agentList.length

                        Text {
                            text: selectedAgent >= 0 && selectedAgent < agentList.length ? agentList[selectedAgent].name : ""
                            color: Theme.text; font.pixelSize: 16; font.bold: true
                        }

                        Row {
                            spacing: 12
                            DetailItem { label: "PID"; value: selectedAgent >= 0 && selectedAgent < agentList.length ? agentList[selectedAgent].atype : "" }
                            DetailItem { label: "Status"; value: selectedAgent >= 0 && selectedAgent < agentList.length ? agentList[selectedAgent].status : "" }
                            DetailItem { label: "Memory"; value: selectedAgent >= 0 && selectedAgent < agentList.length ? agentList[selectedAgent].memory : "" }
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                        /* Actions */
                        Row {
                            spacing: 8

                            Rectangle {
                                width: 90; height: 28; radius: Theme.radiusSmall
                                color: Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)
                                Text { anchors.centerIn: parent; text: "Kill Process"; color: Theme.error; font.pixelSize: 10 }
                                MouseArea {
                                    anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (selectedAgent >= 0 && selectedAgent < agentList.length) {
                                            ProcessManager.killProcess(agentList[selectedAgent].pid)
                                            selectedAgent = -1
                                            refreshAgents()
                                        }
                                    }
                                }
                            }
                        }

                        /* System info log */
                        Rectangle {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            radius: Theme.radius; color: Theme.darkMode ? "#0C0C0C" : "#F5F5F5"

                            Text {
                                anchors.fill: parent; anchors.margins: 8
                                text: "[SYS] Processes: " + ProcessManager.processCount +
                                      "\n[SYS] CPU: " + Math.round(SystemInfo.cpuUsage) + "%" +
                                      "\n[SYS] Memory: " + (SystemInfo.memoryUsed / 1048576).toFixed(0) + " / " + (SystemInfo.memoryTotal / 1048576).toFixed(0) + " MB" +
                                      "\n[SYS] Temp: " + SystemInfo.cpuTemp.toFixed(0) + "\u00B0C" +
                                      "\n[SYS] NPU devices: " + NPUMonitor.deviceCount +
                                      "\n[NPIE] Backend: " + NPIE.currentBackend +
                                      "\n[NPIE] Model: " + (NPIE.modelLoaded ? NPIE.modelName : "none") +
                                      "\n[NPIE] Inferences: " + NPIE.inferenceCount
                                color: Theme.darkMode ? "#10B981" : "#059669"; font.pixelSize: 10; font.family: "monospace"
                                wrapMode: Text.Wrap
                            }
                        }
                    }

                    /* Empty state */
                    Text {
                        anchors.centerIn: parent
                        text: "Select a process to view details"
                        color: Theme.textDim; font.pixelSize: 12
                        visible: selectedAgent < 0 || selectedAgent >= agentList.length
                    }
                }
            }
        }
    }

    component StatPill: Row {
        spacing: 4
        property string label: ""
        property string value: ""
        property color pillColor: Theme.primary

        Text { text: label + ":"; color: Theme.textDim; font.pixelSize: 10; anchors.verticalCenter: parent.verticalCenter }
        Text { text: value; color: pillColor; font.pixelSize: 10; font.bold: true; anchors.verticalCenter: parent.verticalCenter }
    }

    component DetailItem: Column {
        spacing: 2
        property string label: ""
        property string value: ""
        Text { text: label; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
        Text { text: value; color: Theme.text; font.pixelSize: 11 }
    }
}
