import QtQuick 2.15
import QtQuick.Layouts 1.15
import "."

Rectangle {
    color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.95)

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        spacing: 12

        /* Logo + Name */
        Rectangle {
            width: 28; height: 28
            radius: 6
            gradient: Gradient {
                GradientStop { position: 0.0; color: Theme.primary }
                GradientStop { position: 1.0; color: Theme.secondary }
            }

            Text {
                anchors.centerIn: parent
                text: "N"
                color: "#FFFFFF"
                font.pixelSize: 16
                font.bold: true
            }
        }

        Text {
            text: "NeuralOS"
            color: Theme.text
            font.pixelSize: Theme.fontSizeLarge
            font.bold: true
        }

        Item { Layout.fillWidth: true }

        /* System Metrics Mini */
        Row {
            spacing: 16

            /* CPU */
            Row {
                spacing: 4
                Rectangle {
                    width: 8; height: 8; radius: 4
                    anchors.verticalCenter: parent.verticalCenter
                    color: SystemInfo.cpuUsage > 80 ? Theme.error :
                           SystemInfo.cpuUsage > 50 ? Theme.warning : Theme.success
                }
                Text {
                    text: "CPU " + Math.round(SystemInfo.cpuUsage) + "%"
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeSmall
                }
            }

            /* Memory */
            Row {
                spacing: 4
                Rectangle {
                    width: 8; height: 8; radius: 4
                    anchors.verticalCenter: parent.verticalCenter
                    property double memPct: SystemInfo.memoryTotal > 0 ?
                        SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
                    color: memPct > 80 ? Theme.error :
                           memPct > 50 ? Theme.warning : Theme.success
                }
                Text {
                    text: "MEM " + (SystemInfo.memoryTotal > 0 ?
                        Math.round(SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100) : 0) + "%"
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeSmall
                }
            }

            /* NPU */
            Row {
                spacing: 4
                visible: NPUMonitor.deviceCount > 0
                Rectangle {
                    width: 8; height: 8; radius: 4
                    anchors.verticalCenter: parent.verticalCenter
                    color: Theme.primary
                }
                Text {
                    text: "NPU"
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeSmall
                }
            }
        }

        /* Clock */
        Text {
            id: clockText
            color: Theme.text
            font.pixelSize: Theme.fontSizeLarge
            font.bold: true

            Timer {
                interval: 1000
                running: true
                repeat: true
                triggeredOnStart: true
                onTriggered: {
                    var d = new Date()
                    clockText.text = Qt.formatTime(d, "HH:mm")
                }
            }
        }
    }
}
