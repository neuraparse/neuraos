import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: npuApp
    anchors.fill: parent

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        Flickable {
            anchors.fill: parent; anchors.margins: 12
            contentHeight: npuCol.height; clip: true

            ColumnLayout {
                id: npuCol
                width: parent.width
                spacing: 10

                /* Header metrics */
                RowLayout {
                    Layout.fillWidth: true; spacing: 10

                    Components.MetricCard {
                        Layout.fillWidth: true; title: "Devices"; value: NPUMonitor.deviceCount.toString()
                        icon: "\u2756"; accentColor: Theme.secondary
                    }
                    Components.MetricCard {
                        Layout.fillWidth: true; title: "Frequency"; value: NPUMonitor.frequencyMhz + " MHz"
                        icon: "\u2261"; accentColor: Theme.primary
                    }
                    Components.MetricCard {
                        Layout.fillWidth: true; title: "Power"; value: NPUMonitor.powerMw + " mW"
                        icon: "\u26A1"; accentColor: Theme.warning
                    }
                    Components.MetricCard {
                        Layout.fillWidth: true; title: "Inferences"; value: NPUMonitor.totalInferences.toString()
                        icon: "\u2699"; accentColor: Theme.success
                    }
                }

                /* Device list */
                Repeater {
                    model: NPUMonitor.devices

                    Rectangle {
                        Layout.fillWidth: true; height: 100
                        radius: Theme.radius; color: Theme.surface

                        ColumnLayout {
                            anchors.fill: parent; anchors.margins: 12; spacing: 8

                            RowLayout {
                                spacing: 10

                                Rectangle {
                                    width: 40; height: 40; radius: 20
                                    color: Qt.rgba(Theme.secondary.r, Theme.secondary.g, Theme.secondary.b, 0.15)
                                    Text { anchors.centerIn: parent; text: "\u2756"; font.pixelSize: 18; color: Theme.secondary }
                                }

                                Column {
                                    Layout.fillWidth: true; spacing: 2
                                    Text { text: modelData.name; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                    Text { text: modelData.type + " | " + modelData.cores + " cores | " + modelData.memoryMB + " MB SRAM"; color: Theme.textDim; font.pixelSize: 10 }
                                }

                                Components.StatusBadge { text: "Active"; badgeColor: Theme.success }
                            }

                            /* Utilization bar */
                            RowLayout {
                                Layout.fillWidth: true; spacing: 8

                                Text { text: "Utilization"; color: Theme.textDim; font.pixelSize: 10 }

                                Rectangle {
                                    Layout.fillWidth: true; height: 6; radius: 3
                                    color: Theme.surfaceLight
                                    Rectangle { width: parent.width * 0.67; height: parent.height; radius: 3; color: Theme.secondary }
                                }

                                Text { text: "67%"; color: Theme.secondary; font.pixelSize: 10; font.bold: true }
                            }
                        }
                    }
                }

                /* No device fallback */
                Rectangle {
                    Layout.fillWidth: true; height: 60
                    radius: Theme.radius; color: Theme.surface
                    visible: NPUMonitor.deviceCount === 0
                    Text { anchors.centerIn: parent; text: "No NPU devices detected. Using CPU fallback."; color: Theme.textDim; font.pixelSize: 12 }
                }

                /* Frequency control */
                Rectangle {
                    Layout.fillWidth: true; height: 80
                    radius: Theme.radius; color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        RowLayout {
                            Text { text: "Frequency Control"; color: Theme.text; font.pixelSize: 12; font.bold: true }
                            Item { Layout.fillWidth: true }
                            Text { text: NPUMonitor.frequencyMhz + " MHz"; color: Theme.secondary; font.pixelSize: 12; font.bold: true }
                        }

                        Row {
                            spacing: 8

                            Repeater {
                                model: [200, 400, 600, 800, 1000]

                                Rectangle {
                                    width: 70; height: 28; radius: Theme.radiusSmall
                                    color: NPUMonitor.frequencyMhz === modelData ? Theme.secondary : fqMa.containsMouse ? Theme.surfaceAlt : Theme.surface
                                    border.width: 1; border.color: NPUMonitor.frequencyMhz === modelData ? Theme.secondary : Theme.surfaceLight

                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData + " MHz"
                                        color: NPUMonitor.frequencyMhz === modelData ? "#FFF" : Theme.text
                                        font.pixelSize: 10
                                    }
                                    MouseArea {
                                        id: fqMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: NPUMonitor.setFrequency(modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                /* Power management */
                Rectangle {
                    Layout.fillWidth: true; height: 56
                    radius: Theme.radius; color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.margins: 12

                        Column {
                            Layout.fillWidth: true; spacing: 2
                            Text { text: "Power Management"; color: Theme.text; font.pixelSize: 12; font.bold: true }
                            Text { text: "Enable/disable NPU power"; color: Theme.textDim; font.pixelSize: 10 }
                        }

                        Components.ToggleSwitch {
                            checked: true
                            onToggled: NPUMonitor.setPower(checked)
                        }
                    }
                }

                /* Benchmark button */
                Rectangle {
                    Layout.alignment: Qt.AlignHCenter
                    width: 160; height: 36; radius: Theme.radiusSmall
                    color: benchMa.containsMouse ? Qt.darker(Theme.secondary, 1.2) : Theme.secondary

                    Text { anchors.centerIn: parent; text: "Run Benchmark"; color: "#FFF"; font.bold: true; font.pixelSize: 11 }
                    MouseArea {
                        id: benchMa; anchors.fill: parent
                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    }
                }

                Item { height: 8 }
            }
        }
    }
}
