import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15
import ".."
import "../components" as Components

Item {
    Flickable {
        anchors.fill: parent
        anchors.margins: 20
        contentHeight: col.height
        clip: true

        ColumnLayout {
            id: col
            width: parent.width
            spacing: 16

            Components.SectionHeader { title: "System Monitor" }

            /* CPU & Memory Gauges */
            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                /* CPU Card */
                Rectangle {
                    Layout.fillWidth: true
                    height: 200
                    radius: Theme.radius
                    color: Theme.surface

                    Column {
                        anchors.centerIn: parent
                        spacing: 8

                        Components.CircularGauge {
                            width: 120; height: 120
                            anchors.horizontalCenter: parent.horizontalCenter
                            value: SystemInfo.cpuUsage
                            gaugeColor: Theme.primary
                            label: "CPU Usage"
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "Temp: " + SystemInfo.cpuTemp.toFixed(1) + "\u00B0C"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontSizeSmall
                        }
                    }
                }

                /* Memory Card */
                Rectangle {
                    Layout.fillWidth: true
                    height: 200
                    radius: Theme.radius
                    color: Theme.surface

                    Column {
                        anchors.centerIn: parent
                        spacing: 8

                        Components.CircularGauge {
                            width: 120; height: 120
                            anchors.horizontalCenter: parent.horizontalCenter
                            value: SystemInfo.memoryTotal > 0 ?
                                SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
                            gaugeColor: Theme.success
                            label: "Memory"
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: (SystemInfo.memoryUsed / 1048576).toFixed(0) + " / " +
                                  (SystemInfo.memoryTotal / 1048576).toFixed(0) + " MB"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontSizeSmall
                        }
                    }
                }

                /* Disk Card */
                Rectangle {
                    Layout.fillWidth: true
                    height: 200
                    radius: Theme.radius
                    color: Theme.surface

                    Column {
                        anchors.centerIn: parent
                        spacing: 8

                        Components.CircularGauge {
                            width: 120; height: 120
                            anchors.horizontalCenter: parent.horizontalCenter
                            value: SystemInfo.diskTotal > 0 ?
                                SystemInfo.diskUsed / SystemInfo.diskTotal * 100 : 0
                            gaugeColor: Theme.warning
                            label: "Disk"
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: (SystemInfo.diskUsed / 1073741824).toFixed(1) + " / " +
                                  (SystemInfo.diskTotal / 1073741824).toFixed(1) + " GB"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontSizeSmall
                        }
                    }
                }
            }

            /* CPU History Chart (Canvas) */
            Components.SectionHeader { title: "CPU History (60s)" }

            Rectangle {
                Layout.fillWidth: true
                height: 150
                radius: Theme.radius
                color: Theme.surface

                Canvas {
                    id: cpuChart
                    anchors.fill: parent
                    anchors.margins: 12

                    property var history: SystemInfo.cpuHistory

                    onHistoryChanged: requestPaint()

                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.clearRect(0, 0, width, height)

                        var data = history
                        if (data.length < 2) return

                        var step = width / 59
                        ctx.beginPath()
                        ctx.moveTo(0, height - (data[0] / 100 * height))
                        for (var i = 1; i < data.length; i++) {
                            ctx.lineTo(i * step, height - (data[i] / 100 * height))
                        }
                        ctx.strokeStyle = Theme.primary
                        ctx.lineWidth = 2
                        ctx.stroke()

                        /* Fill */
                        ctx.lineTo((data.length - 1) * step, height)
                        ctx.lineTo(0, height)
                        ctx.closePath()
                        ctx.fillStyle = Qt.rgba(0, 0.85, 1, 0.1)
                        ctx.fill()
                    }
                }
            }

            /* Process List */
            Components.SectionHeader { title: "Processes (" + ProcessManager.processCount + ")" }

            Rectangle {
                Layout.fillWidth: true
                height: Math.min(ProcessManager.processCount * 32 + 40, 300)
                radius: Theme.radius
                color: Theme.surface
                clip: true

                Column {
                    anchors.fill: parent
                    anchors.margins: 8

                    /* Header */
                    Row {
                        width: parent.width
                        height: 28
                        spacing: 0

                        Text { width: 60;  text: "PID"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 200; text: "Name"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 60;  text: "State"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 80;  text: "RSS (KB)"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                    }

                    ListView {
                        width: parent.width
                        height: parent.height - 32
                        model: ProcessManager.processes
                        clip: true

                        delegate: Row {
                            width: parent ? parent.width : 0
                            height: 28
                            spacing: 0

                            Text { width: 60;  text: modelData.pid || ""; color: Theme.text; font.pixelSize: 11 }
                            Text { width: 200; text: modelData.name || ""; color: Theme.text; font.pixelSize: 11; elide: Text.ElideRight }
                            Text { width: 60;  text: modelData.state || ""; color: Theme.textDim; font.pixelSize: 11 }
                            Text { width: 80;  text: modelData.rss || ""; color: Theme.textDim; font.pixelSize: 11 }
                        }
                    }
                }
            }

            Rectangle {
                Layout.alignment: Qt.AlignRight
                width: 100; height: 32
                radius: Theme.radiusSmall
                color: Theme.surfaceAlt

                Text {
                    anchors.centerIn: parent
                    text: "Refresh"
                    color: Theme.primary
                    font.pixelSize: Theme.fontSizeSmall
                }

                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: ProcessManager.refresh()
                }
            }

            Item { height: 20 }
        }
    }
}
