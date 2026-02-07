import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: ecoApp
    anchors.fill: parent

    property int selectedDevice: -1

    function getDeviceDetail() {
        if (selectedDevice < 0) return null
        return Ecosystem.getDeviceDetail(selectedDevice)
    }

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

                    Text { text: "Ecosystem"; color: Theme.ecosystem; font.pixelSize: 15; font.bold: true }

                    Item { Layout.fillWidth: true }

                    /* Sync status */
                    Rectangle {
                        width: syncRow.width + 16; height: 26; radius: 13
                        color: Ecosystem.syncStatus === "idle" ?
                            Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15) :
                            Qt.rgba(Theme.warning.r, Theme.warning.g, Theme.warning.b, 0.15)
                        Row {
                            id: syncRow; anchors.centerIn: parent; spacing: 6
                            Rectangle { width: 8; height: 8; radius: 4; color: Ecosystem.syncStatus === "idle" ? Theme.success : Theme.warning }
                            Text {
                                text: Ecosystem.syncStatus === "idle" ? "Synced" : Ecosystem.syncStatus
                                color: Ecosystem.syncStatus === "idle" ? Theme.success : Theme.warning
                                font.pixelSize: 10; font.bold: true
                            }
                        }
                    }

                    Row {
                        spacing: 12
                        StatPill { label: "Devices"; value: Ecosystem.totalDevices.toString(); pillColor: Theme.ecosystem }
                        StatPill { label: "Connected"; value: Ecosystem.connectedCount.toString(); pillColor: Theme.success }
                    }
                }
            }

            /* ─── Content ─── */
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; spacing: 0

                /* ─── Device List (left) ─── */
                Rectangle {
                    Layout.preferredWidth: 300; Layout.fillHeight: true; color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        Text { text: "Edge Devices"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: Ecosystem.devices; clip: true; spacing: 4

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: 72; radius: Theme.radiusTiny
                                color: selectedDevice === modelData.id ? Theme.glassActive :
                                       devMa.containsMouse ? Theme.glassHover : "transparent"

                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 3

                                    RowLayout {
                                        spacing: 8
                                        Rectangle {
                                            width: 28; height: 28; radius: Theme.radiusTiny
                                            color: Qt.rgba(Theme.ecosystem.r, Theme.ecosystem.g, Theme.ecosystem.b, 0.15)
                                            Text {
                                                anchors.centerIn: parent
                                                text: {
                                                    var t = modelData.type
                                                    if (t === "compute") return "\u{1F5A5}"
                                                    if (t === "sensor") return "\u{1F4E1}"
                                                    if (t === "gateway") return "\u{1F310}"
                                                    if (t === "edge") return "\u2B22"
                                                    return "\u25CF"
                                                }
                                                font.pixelSize: 13
                                            }
                                        }

                                        ColumnLayout {
                                            spacing: 1; Layout.fillWidth: true
                                            Text {
                                                text: modelData.name; color: Theme.text
                                                font.pixelSize: 12; font.bold: true
                                                Layout.fillWidth: true; elide: Text.ElideRight
                                            }
                                            Text {
                                                text: modelData.ip + " | " + modelData.arch
                                                color: Theme.textDim; font.pixelSize: 10
                                            }
                                        }

                                        Rectangle {
                                            width: 10; height: 10; radius: 5
                                            color: modelData.status === "connected" ? Theme.success :
                                                   modelData.status === "syncing" ? Theme.warning : Theme.textMuted
                                        }
                                    }

                                    /* Mini stats bar */
                                    RowLayout {
                                        spacing: 8
                                        visible: modelData.status === "connected"

                                        Rectangle {
                                            width: 50; height: 4; radius: 2; color: Theme.surfaceLight
                                            Rectangle {
                                                width: parent.width * (modelData.cpuUsage / 100); height: 4; radius: 2
                                                color: modelData.cpuUsage > 80 ? Theme.error : modelData.cpuUsage > 50 ? Theme.warning : Theme.success
                                            }
                                        }
                                        Text { text: "CPU " + modelData.cpuUsage.toFixed(0) + "%"; color: Theme.textMuted; font.pixelSize: 9 }

                                        Rectangle {
                                            width: 50; height: 4; radius: 2; color: Theme.surfaceLight
                                            Rectangle {
                                                width: parent.width * (modelData.memoryUsage / 100); height: 4; radius: 2
                                                color: modelData.memoryUsage > 80 ? Theme.error : Theme.primary
                                            }
                                        }
                                        Text { text: "MEM " + modelData.memoryUsage.toFixed(0) + "%"; color: Theme.textMuted; font.pixelSize: 9 }
                                    }
                                }

                                MouseArea {
                                    id: devMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedDevice = modelData.id
                                }
                            }
                        }

                        /* Scan button */
                        Rectangle {
                            Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                            color: scanMa.containsMouse ? Qt.darker(Theme.ecosystem, 1.2) : Theme.ecosystem

                            Text { anchors.centerIn: parent; text: "Scan Network"; color: "white"; font.pixelSize: 12; font.bold: true }
                            MouseArea {
                                id: scanMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: Ecosystem.scanDevices()
                            }
                        }
                    }
                }

                Rectangle { Layout.fillHeight: true; width: 1; color: Theme.glassBorder }

                /* ─── Device Detail (right) ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true; color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 16

                        property var detail: getDeviceDetail()

                        Text {
                            text: parent.detail ? parent.detail.name : "Select a Device"
                            color: Theme.text; font.pixelSize: 16; font.bold: true
                        }

                        /* Device info grid */
                        GridLayout {
                            Layout.fillWidth: true; columns: 3; rowSpacing: 12; columnSpacing: 12
                            visible: parent.detail !== null

                            Repeater {
                                model: {
                                    var d = parent.parent.detail
                                    if (!d) return []
                                    return [
                                        { label: "IP Address", value: d.ip, clr: Theme.ecosystem },
                                        { label: "Architecture", value: d.arch, clr: Theme.primary },
                                        { label: "Type", value: d.type, clr: Theme.secondary },
                                        { label: "CPU Usage", value: d.cpuUsage.toFixed(1) + "%", clr: d.cpuUsage > 80 ? Theme.error : Theme.success },
                                        { label: "Memory", value: d.memoryUsage.toFixed(1) + "%", clr: Theme.primary },
                                        { label: "Models", value: d.loadedModels.toString(), clr: Theme.warning },
                                        { label: "Active Tasks", value: d.activeTasks.toString(), clr: Theme.aiBus },
                                        { label: "Status", value: d.status, clr: d.status === "connected" ? Theme.success : Theme.error }
                                    ]
                                }

                                Rectangle {
                                    Layout.fillWidth: true; height: 70; radius: Theme.radiusSmall
                                    color: Theme.surface; border.width: 1; border.color: Theme.glassBorder

                                    ColumnLayout {
                                        anchors.centerIn: parent; spacing: 4
                                        Text {
                                            text: modelData.value; color: modelData.clr
                                            font.pixelSize: 18; font.bold: true
                                            Layout.alignment: Qt.AlignHCenter
                                        }
                                        Text {
                                            text: modelData.label; color: Theme.textDim; font.pixelSize: 10
                                            Layout.alignment: Qt.AlignHCenter
                                        }
                                    }
                                }
                            }
                        }

                        /* Action buttons */
                        RowLayout {
                            Layout.fillWidth: true; spacing: 8
                            visible: parent.detail !== null

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                                color: {
                                    var d = parent.parent.detail
                                    if (!d) return Theme.surfaceAlt
                                    return connectMa.containsMouse ?
                                        (d.status === "connected" ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.3) : Qt.darker(Theme.success, 1.2)) :
                                        (d.status === "connected" ? Theme.surfaceAlt : Theme.success)
                                }
                                border.width: { var d = parent.parent.detail; return d && d.status === "connected" ? 1 : 0 }
                                border.color: Theme.error

                                Text {
                                    anchors.centerIn: parent
                                    text: { var d = parent.parent.parent.detail; return d && d.status === "connected" ? "Disconnect" : "Connect" }
                                    color: { var d = parent.parent.parent.detail; return d && d.status === "connected" ? Theme.error : "white" }
                                    font.pixelSize: 12; font.bold: true
                                }
                                MouseArea {
                                    id: connectMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        var d = getDeviceDetail()
                                        if (!d) return
                                        if (d.status === "connected") Ecosystem.disconnectDevice(d.id)
                                        else Ecosystem.connectDevice(d.ip)
                                    }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                                color: syncModelsMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt

                                Text { anchors.centerIn: parent; text: "Sync Models"; color: Theme.text; font.pixelSize: 12 }
                                MouseArea {
                                    id: syncModelsMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: if (selectedDevice > 0) Ecosystem.syncModels(selectedDevice)
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                                color: distTaskMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt

                                Text { anchors.centerIn: parent; text: "Distribute Task"; color: Theme.text; font.pixelSize: 12 }
                                MouseArea {
                                    id: distTaskMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: if (selectedDevice > 0) Ecosystem.distributeTask("inference_task", selectedDevice)
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }
            }
        }
    }

    Timer {
        interval: 3000; running: true; repeat: true
        onTriggered: Ecosystem.refresh()
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
