import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: mcpApp
    anchors.fill: parent

    property int activeTab: 0

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Top Bar ─── */
            Rectangle {
                Layout.fillWidth: true; height: 46; color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 10; spacing: 12

                    Text { text: "MCP Hub"; color: Theme.mcp; font.pixelSize: 15; font.bold: true }

                    Item { width: 12 }

                    Repeater {
                        model: ["Server", "Tools", "Logs"]

                        Rectangle {
                            width: 72; height: 30; radius: Theme.radiusSmall
                            color: activeTab === index ?
                                Qt.rgba(Theme.mcp.r, Theme.mcp.g, Theme.mcp.b, 0.2) :
                                mcpTabMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Text {
                                anchors.centerIn: parent; text: modelData
                                color: activeTab === index ? Theme.mcp : Theme.text
                                font.pixelSize: 11; font.bold: activeTab === index
                            }
                            MouseArea {
                                id: mcpTabMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: activeTab = index
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Server status indicator */
                    Rectangle {
                        width: srvRow.width + 16; height: 26; radius: 13
                        color: MCP.serverStatus === "running" ?
                            Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15) :
                            Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)

                        Row {
                            id: srvRow; anchors.centerIn: parent; spacing: 6
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: MCP.serverStatus === "running" ? Theme.success : Theme.error
                            }
                            Text {
                                text: MCP.serverStatus === "running" ? "Port " + MCP.serverPort : "Stopped"
                                color: MCP.serverStatus === "running" ? Theme.success : Theme.error
                                font.pixelSize: 10; font.bold: true
                            }
                        }
                    }

                    Row {
                        spacing: 12
                        StatPill { label: "Clients"; value: MCP.connectedClients.toString(); pillColor: Theme.mcp }
                        StatPill { label: "Tools"; value: MCP.tools.length.toString(); pillColor: Theme.primary }
                    }
                }
            }

            /* ─── Content ─── */
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; spacing: 0

                /* ─── Left Panel ─── */
                Rectangle {
                    Layout.preferredWidth: 300; Layout.fillHeight: true; color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        /* Server controls */
                        Text { text: "MCP Server"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 8

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                                color: startMa.containsMouse ? Qt.darker(Theme.mcp, 1.2) : Theme.mcp
                                visible: MCP.serverStatus !== "running"

                                Text { anchors.centerIn: parent; text: "Start Server"; color: "white"; font.pixelSize: 12; font.bold: true }
                                MouseArea {
                                    id: startMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: MCP.startServer(3100)
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                                color: stopMa.containsMouse ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.3) : Theme.surfaceAlt
                                visible: MCP.serverStatus === "running"
                                border.width: 1; border.color: Theme.error

                                Text { anchors.centerIn: parent; text: "Stop Server"; color: Theme.error; font.pixelSize: 12; font.bold: true }
                                MouseArea {
                                    id: stopMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: MCP.stopServer()
                                }
                            }
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.glassBorder }

                        /* Connected Clients */
                        Text { text: "Connected Clients"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: MCP.clients; clip: true; spacing: 4

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: 52; radius: Theme.radiusTiny
                                color: clientMa.containsMouse ? Theme.glassHover : "transparent"

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 10

                                    Rectangle {
                                        width: 32; height: 32; radius: Theme.radiusTiny
                                        color: Qt.rgba(Theme.mcp.r, Theme.mcp.g, Theme.mcp.b, 0.15)
                                        Text {
                                            anchors.centerIn: parent; text: "\u26A1"
                                            color: Theme.mcp; font.pixelSize: 14
                                        }
                                    }

                                    ColumnLayout {
                                        spacing: 2; Layout.fillWidth: true
                                        Text {
                                            text: modelData.name; color: Theme.text
                                            font.pixelSize: 12; font.bold: true
                                        }
                                        Text {
                                            text: modelData.type + " | " + modelData.connectedAt
                                            color: Theme.textDim; font.pixelSize: 10
                                        }
                                    }

                                    Rectangle {
                                        width: 8; height: 8; radius: 4
                                        color: modelData.status === "connected" ? Theme.success : Theme.error
                                    }
                                }

                                MouseArea {
                                    id: clientMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                }
                            }

                            Text {
                                anchors.centerIn: parent
                                text: MCP.serverStatus === "running" ? "No clients connected" : "Start server to accept connections"
                                color: Theme.textMuted; font.pixelSize: 12
                                visible: MCP.clients.length === 0
                            }
                        }
                    }
                }

                Rectangle { Layout.fillHeight: true; width: 1; color: Theme.glassBorder }

                /* ─── Right Panel (Tools / Logs) ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true; color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 12

                        /* Tools view */
                        Text {
                            text: activeTab === 2 ? "Server Logs" : "Registered Tools"
                            color: Theme.text; font.pixelSize: 16; font.bold: true
                        }

                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: activeTab === 2 ? MCP.logs : MCP.tools
                            clip: true; spacing: 4
                            visible: true

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: activeTab === 2 ? 32 : 60
                                radius: Theme.radiusTiny
                                color: toolMa.containsMouse ? Theme.glassHover : Theme.surface
                                border.width: activeTab === 2 ? 0 : 1
                                border.color: Theme.glassBorder

                                /* Tool view */
                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 2
                                    visible: activeTab !== 2

                                    RowLayout {
                                        spacing: 8
                                        Text {
                                            text: modelData.name || ""; color: Theme.mcp
                                            font.pixelSize: 12; font.bold: true
                                            Layout.fillWidth: true
                                        }
                                        Rectangle {
                                            width: enabledBadge.width + 12; height: 18; radius: 9
                                            color: (modelData.enabled !== undefined && modelData.enabled) ?
                                                Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15) :
                                                Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)
                                            Text {
                                                id: enabledBadge; anchors.centerIn: parent
                                                text: (modelData.enabled !== undefined && modelData.enabled) ? "Enabled" : "Disabled"
                                                color: (modelData.enabled !== undefined && modelData.enabled) ? Theme.success : Theme.error
                                                font.pixelSize: 9; font.bold: true
                                            }
                                        }
                                    }
                                    Text {
                                        text: modelData.description || ""
                                        color: Theme.textDim; font.pixelSize: 10
                                        Layout.fillWidth: true; elide: Text.ElideRight
                                    }
                                }

                                /* Log view */
                                Text {
                                    anchors.verticalCenter: parent.verticalCenter; x: 10
                                    text: typeof modelData === "string" ? modelData : (modelData.toString ? modelData.toString() : "")
                                    color: Theme.textDim; font.pixelSize: 11
                                    font.family: "monospace"
                                    visible: activeTab === 2
                                }

                                MouseArea {
                                    id: toolMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }

                        /* Clear logs button */
                        Rectangle {
                            Layout.fillWidth: true; height: 32; radius: Theme.radiusSmall
                            color: clearLogsMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt
                            visible: activeTab === 2

                            Text { anchors.centerIn: parent; text: "Clear Logs"; color: Theme.text; font.pixelSize: 11 }
                            MouseArea {
                                id: clearLogsMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: MCP.clearLogs()
                            }
                        }
                    }
                }
            }
        }
    }

    component StatPill: Rectangle {
        property string label: ""; property string value: ""; property color pillColor: Theme.primary
        width: spRow.width + 16; height: 24; radius: 12
        color: Qt.rgba(pillColor.r, pillColor.g, pillColor.b, 0.12)
        Row { id: spRow; anchors.centerIn: parent; spacing: 4
            Text { text: parent.parent.label; color: Theme.textDim; font.pixelSize: 10 }
            Text { text: parent.parent.value; color: parent.parent.pillColor; font.pixelSize: 10; font.bold: true }
        }
    }
}
