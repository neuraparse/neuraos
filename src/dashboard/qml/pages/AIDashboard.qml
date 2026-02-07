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

            Components.SectionHeader { title: "AI Inference Engine" }

            /* Top metrics row */
            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Backend"
                    value: NPIE.currentBackend
                    icon: "\u2699"
                    accentColor: Theme.primary
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Inferences"
                    value: NPIE.inferenceCount.toString()
                    unit: "total"
                    icon: "\u2261"
                    accentColor: Theme.success
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Last Inference"
                    value: NPIE.lastInferenceMs.toFixed(2)
                    unit: "ms"
                    icon: "\u23F1"
                    accentColor: Theme.warning
                }
                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Avg Inference"
                    value: NPIE.avgInferenceMs.toFixed(2)
                    unit: "ms"
                    icon: "\u2248"
                    accentColor: Theme.secondary
                }
            }

            /* Backend Selection */
            Components.SectionHeader { title: "Backend Selection" }

            Rectangle {
                Layout.fillWidth: true
                height: 50
                radius: Theme.radius
                color: Theme.surface

                Row {
                    anchors.centerIn: parent
                    spacing: 8

                    Repeater {
                        model: NPIE.backends

                        delegate: Rectangle {
                            width: 100; height: 34
                            radius: Theme.radiusSmall
                            color: NPIE.currentBackend === modelData ?
                                Theme.primary : Theme.surfaceAlt
                            border.width: 1
                            border.color: NPIE.currentBackend === modelData ?
                                Theme.primary : Theme.surfaceLight

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                color: NPIE.currentBackend === modelData ? "#000000" : Theme.text
                                font.pixelSize: Theme.fontSizeSmall
                                font.bold: NPIE.currentBackend === modelData
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: NPIE.setBackend(modelData)
                            }
                        }
                    }
                }
            }

            /* Model Controls */
            Components.SectionHeader { title: "Model" }

            Rectangle {
                Layout.fillWidth: true
                height: 80
                radius: Theme.radius
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    Column {
                        Layout.fillWidth: true
                        spacing: 4
                        Text {
                            text: NPIE.modelLoaded ? NPIE.modelName : "No model loaded"
                            color: Theme.text
                            font.pixelSize: Theme.fontSizeNormal
                        }
                        Components.StatusBadge {
                            text: NPIE.modelLoaded ? "Loaded" : "Idle"
                            badgeColor: NPIE.modelLoaded ? Theme.success : Theme.textDim
                        }
                    }

                    Rectangle {
                        width: 120; height: 36
                        radius: Theme.radiusSmall
                        color: Theme.primary

                        Text {
                            anchors.centerIn: parent
                            text: NPIE.modelLoaded ? "Run Inference" : "Load Demo"
                            color: "#000000"
                            font.bold: true
                            font.pixelSize: Theme.fontSizeSmall
                        }

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                if (NPIE.modelLoaded) {
                                    NPIE.runInference()
                                } else {
                                    NPIE.loadModel("/opt/models/demo_model.tflite")
                                }
                            }
                        }
                    }
                }
            }

            /* NPU Section */
            Components.SectionHeader { title: "NPU Devices"; accentColor: Theme.secondary }

            Repeater {
                model: NPUMonitor.devices

                delegate: Rectangle {
                    Layout.fillWidth: true
                    height: 80
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 16

                        Rectangle {
                            width: 48; height: 48; radius: 24
                            color: Qt.rgba(Theme.secondary.r, Theme.secondary.g, Theme.secondary.b, 0.15)
                            Text {
                                anchors.centerIn: parent
                                text: "\u2756"
                                font.pixelSize: 22
                                color: Theme.secondary
                            }
                        }

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: modelData.name; color: Theme.text; font.pixelSize: Theme.fontSizeNormal; font.bold: true }
                            Text { text: modelData.type + " | " + modelData.cores + " cores | " + modelData.memoryMB + " MB"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Column {
                            spacing: 2
                            Text { text: NPUMonitor.totalInferences + " inferences"; color: Theme.primary; font.pixelSize: Theme.fontSizeSmall }
                            Text { text: NPUMonitor.powerMw + " mW"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }
                    }
                }
            }

            Item { height: 20 }
        }
    }
}
