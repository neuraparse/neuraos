import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: busApp
    anchors.fill: parent

    property int selectedAgent: -1
    property int activeTab: 0

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Tab Bar ─── */
            Rectangle {
                Layout.fillWidth: true
                height: 42
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 8; spacing: 8

                    Text { text: "AI Bus"; color: Theme.aiBus; font.pixelSize: 15; font.bold: true }

                    Item { width: 16 }

                    Repeater {
                        model: ["Agents", "Pipelines", "Metrics"]

                        Rectangle {
                            width: 90; height: 30; radius: Theme.radiusSmall
                            color: activeTab === index ?
                                Qt.rgba(Theme.aiBus.r, Theme.aiBus.g, Theme.aiBus.b, 0.2) :
                                busTabMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                color: activeTab === index ? Theme.aiBus : Theme.text
                                font.pixelSize: 11; font.bold: activeTab === index
                            }
                            MouseArea {
                                id: busTabMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: activeTab = index
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Stats pills */
                    Row {
                        spacing: 12
                        StatPill { label: "Agents"; value: AIBus.agentCount.toString(); pillColor: Theme.aiBus }
                        StatPill { label: "Active"; value: AIBus.activeAgents.toString(); pillColor: Theme.success }
                        StatPill { label: "Pipelines"; value: AIBus.pipelineCount.toString(); pillColor: Theme.secondary }
                        StatPill { label: "Inferences"; value: AIBus.totalInferences.toString(); pillColor: Theme.warning }
                    }
                }
            }

            /* ─── Content ─── */
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* ─── Agent/Pipeline List (left) ─── */
                Rectangle {
                    Layout.preferredWidth: 280
                    Layout.fillHeight: true
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        Text {
                            text: activeTab === 1 ? "Pipelines" : "Registered Agents"
                            color: Theme.text; font.pixelSize: 13; font.bold: true
                        }

                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: activeTab === 1 ? AIBus.pipelines : AIBus.agents
                            clip: true; spacing: 4

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: 62; radius: Theme.radiusTiny
                                color: {
                                    if (selectedAgent === modelData.id) return Theme.glassActive
                                    if (agentItemMa.containsMouse) return Theme.glassHover
                                    return "transparent"
                                }

                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 2

                                    RowLayout {
                                        spacing: 8
                                        Rectangle {
                                            width: 8; height: 8; radius: 4
                                            color: modelData.status === "running" ? Theme.success :
                                                   modelData.status === "completed" ? Theme.primary :
                                                   modelData.status === "error" ? Theme.error : Theme.textMuted
                                        }
                                        Text {
                                            text: modelData.name; color: Theme.text
                                            font.pixelSize: 12; font.bold: true
                                            Layout.fillWidth: true; elide: Text.ElideRight
                                        }
                                    }

                                    RowLayout {
                                        spacing: 8
                                        Text {
                                            text: activeTab === 1 ?
                                                (modelData.agentIds ? modelData.agentIds.length + " agents" : "") :
                                                (modelData.model || "")
                                            color: Theme.textDim; font.pixelSize: 10
                                        }
                                        Item { Layout.fillWidth: true }
                                        Text {
                                            text: activeTab === 1 ?
                                                (modelData.lastRunMs ? modelData.lastRunMs.toFixed(1) + " ms" : "—") :
                                                (modelData.backend || "")
                                            color: Theme.textDim; font.pixelSize: 10
                                        }
                                    }
                                }

                                MouseArea {
                                    id: agentItemMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedAgent = modelData.id
                                }
                            }
                        }

                        /* Add button */
                        Rectangle {
                            Layout.fillWidth: true; height: 36
                            radius: Theme.radiusSmall
                            color: addBtnMa.containsMouse ? Qt.darker(Theme.aiBus, 1.2) : Theme.aiBus

                            Text {
                                anchors.centerIn: parent
                                text: activeTab === 1 ? "+ New Pipeline" : "+ New Agent"
                                color: "white"; font.pixelSize: 12; font.bold: true
                            }
                            MouseArea {
                                id: addBtnMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    if (activeTab === 1)
                                        AIBus.createPipeline("New Pipeline " + (AIBus.pipelineCount + 1), [1, 2])
                                    else
                                        AIBus.createAgent("New Agent " + (AIBus.agentCount + 1), "custom_model", "auto")
                                }
                            }
                        }
                    }
                }

                /* ─── Separator ─── */
                Rectangle { Layout.fillHeight: true; width: 1; color: Theme.glassBorder }

                /* ─── Detail / Metrics Panel (right) ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 16

                        /* Header */
                        Text {
                            text: activeTab === 2 ? "Bus Metrics" : "Agent Detail"
                            color: Theme.text; font.pixelSize: 16; font.bold: true
                        }

                        /* Metrics Grid */
                        GridLayout {
                            Layout.fillWidth: true
                            columns: 3; rowSpacing: 12; columnSpacing: 12

                            Repeater {
                                model: [
                                    { label: "Total Inferences", value: AIBus.totalInferences.toString(), clr: Theme.aiBus },
                                    { label: "Bus Latency", value: AIBus.busLatencyMs.toFixed(1) + " ms", clr: Theme.warning },
                                    { label: "Active Agents", value: AIBus.activeAgents + " / " + AIBus.agentCount, clr: Theme.success },
                                    { label: "Pipelines", value: AIBus.pipelineCount.toString(), clr: Theme.secondary },
                                    { label: "Throughput", value: (AIBus.totalInferences > 0 ? (AIBus.totalInferences / Math.max(1, AIBus.busLatencyMs) * 1000).toFixed(0) : "0") + " inf/s", clr: Theme.primary },
                                    { label: "Status", value: AIBus.activeAgents > 0 ? "Active" : "Idle", clr: AIBus.activeAgents > 0 ? Theme.success : Theme.textMuted }
                                ]

                                Rectangle {
                                    Layout.fillWidth: true; height: 80
                                    radius: Theme.radiusSmall; color: Theme.surface
                                    border.width: 1; border.color: Theme.glassBorder

                                    ColumnLayout {
                                        anchors.centerIn: parent; spacing: 4
                                        Text {
                                            text: modelData.value; color: modelData.clr
                                            font.pixelSize: 20; font.bold: true
                                            Layout.alignment: Qt.AlignHCenter
                                        }
                                        Text {
                                            text: modelData.label; color: Theme.textDim
                                            font.pixelSize: 11
                                            Layout.alignment: Qt.AlignHCenter
                                        }
                                    }
                                }
                            }
                        }

                        /* Action buttons */
                        RowLayout {
                            Layout.fillWidth: true; spacing: 8

                            Repeater {
                                model: [
                                    { text: "Start All Agents", action: "startAll" },
                                    { text: "Stop All Agents", action: "stopAll" },
                                    { text: "Run Pipeline", action: "runPipeline" },
                                    { text: "Refresh", action: "refresh" }
                                ]

                                Rectangle {
                                    Layout.fillWidth: true; height: 34
                                    radius: Theme.radiusSmall
                                    color: actionMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt

                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData.text; color: Theme.text
                                        font.pixelSize: 11
                                    }
                                    MouseArea {
                                        id: actionMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            if (modelData.action === "startAll") {
                                                var agents = AIBus.agents
                                                for (var i = 0; i < agents.length; i++) AIBus.startAgent(agents[i].id)
                                            } else if (modelData.action === "stopAll") {
                                                var agents2 = AIBus.agents
                                                for (var j = 0; j < agents2.length; j++) AIBus.stopAgent(agents2[j].id)
                                            } else if (modelData.action === "runPipeline") {
                                                var pips = AIBus.pipelines
                                                if (pips.length > 0) AIBus.executePipeline(pips[0].id)
                                            } else if (modelData.action === "refresh") {
                                                AIBus.refresh()
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }
            }
        }
    }

    /* ─── Stat Pill helper ─── */
    component StatPill: Rectangle {
        property string label: ""
        property string value: ""
        property color pillColor: Theme.primary
        width: pillRow.width + 16; height: 24; radius: 12
        color: Qt.rgba(pillColor.r, pillColor.g, pillColor.b, 0.12)
        Row {
            id: pillRow; anchors.centerIn: parent; spacing: 4
            Text { text: parent.parent.label; color: Theme.textDim; font.pixelSize: 10 }
            Text { text: parent.parent.value; color: parent.parent.pillColor; font.pixelSize: 10; font.bold: true }
        }
    }
}
