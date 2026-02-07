import QtQuick 2.15
import QtQuick.Layouts 1.15
import Qt.labs.folderlistmodel 2.15
import ".."
import "../components" as Components

Item {
    id: studioApp
    anchors.fill: parent

    property int activeTab: 0

    FolderListModel {
        id: modelsFolder
        folder: "file:///root/neuraos/models"
        nameFilters: ["*.tflite", "*.onnx", "*.bin", "*.pt"]
        showDirs: false
        sortField: FolderListModel.Name
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B"
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB"
        return (bytes / 1073741824).toFixed(1) + " GB"
    }

    function frameworkFromName(name) {
        var ext = name.split('.').pop().toLowerCase()
        if (ext === "tflite") return "TFLite"
        if (ext === "onnx") return "ONNX"
        if (ext === "pt") return "PyTorch"
        if (ext === "bin") return "Custom"
        return "Unknown"
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Tab bar */
            Rectangle {
                id: nsTabBar
                Layout.fillWidth: true
                height: 38
                color: Theme.surface

                Row {
                    x: 12; y: (nsTabBar.height - height) / 2
                    spacing: 2

                    Repeater {
                        model: ["Overview", "Models", "NPU", "Inference"]

                        Rectangle {
                            width: 90; height: 30
                            radius: Theme.radiusSmall
                            color: activeTab === index ?
                                Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2) :
                                tabMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                color: activeTab === index ? Theme.primary : Theme.text
                                font.pixelSize: 11
                                font.bold: activeTab === index
                            }

                            MouseArea {
                                id: tabMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: activeTab = index
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Content */
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentHeight: contentStack.height
                clip: true

                Column {
                    id: contentStack
                    width: parent.width

                    /* ─── Overview Tab ─── */
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        visible: activeTab === 0

                        Item { height: 12 }

                        /* Top metrics */
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            spacing: 10

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

                        /* Model status + action */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            height: 70
                            radius: Theme.radius
                            color: Theme.surface

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 14; spacing: 14

                                Rectangle {
                                    width: 44; height: 44; radius: 22
                                    color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    Text { anchors.centerIn: parent; text: "\u2756"; font.pixelSize: 22; color: Theme.primary }
                                }

                                Column {
                                    Layout.fillWidth: true; spacing: 2
                                    Text { text: NPIE.modelLoaded ? NPIE.modelName : "No model loaded"; color: Theme.text; font.pixelSize: 12 }
                                    Text {
                                        text: NPIE.modelLoaded ? "Ready for inference" : "Load a model to begin"
                                        color: Theme.textDim; font.pixelSize: 10
                                    }
                                }

                                Components.StatusBadge {
                                    text: NPIE.modelLoaded ? "Loaded" : "Idle"
                                    badgeColor: NPIE.modelLoaded ? Theme.success : Theme.textDim
                                }

                                Rectangle {
                                    width: 110; height: 34; radius: Theme.radiusSmall
                                    color: runMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary

                                    Text {
                                        anchors.centerIn: parent
                                        text: NPIE.modelLoaded ? "Run Inference" : "Load Demo"
                                        color: "#000000"; font.bold: true; font.pixelSize: 11
                                    }

                                    MouseArea {
                                        id: runMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: NPIE.modelLoaded ? NPIE.runInference() : NPIE.loadModel("/opt/models/demo_model.tflite")
                                    }
                                }
                            }
                        }

                        /* Backend selector */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            height: 50
                            radius: Theme.radius
                            color: Theme.surface

                            Row {
                                anchors.centerIn: parent; spacing: 8

                                Text { text: "Backend:"; color: Theme.textDim; font.pixelSize: 11; height: 30; verticalAlignment: Text.AlignVCenter }

                                Repeater {
                                    model: NPIE.backends

                                    Rectangle {
                                        width: 90; height: 30; radius: Theme.radiusSmall
                                        color: NPIE.currentBackend === modelData ? Theme.primary : Theme.surfaceAlt
                                        border.width: 1
                                        border.color: NPIE.currentBackend === modelData ? Theme.primary : Theme.surfaceLight

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData
                                            color: NPIE.currentBackend === modelData ? "#000000" : Theme.text
                                            font.pixelSize: 11; font.bold: NPIE.currentBackend === modelData
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

                        Item { height: 8 }
                    }

                    /* ─── Models Tab ─── */
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        visible: activeTab === 1

                        Item { height: 12 }

                        /* Info */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            height: 28
                            radius: Theme.radiusSmall
                            color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.05)

                            Text {
                                x: 10; y: (parent.height - height) / 2
                                text: modelsFolder.count + " models in /root/neuraos/models"
                                color: Theme.textDim; font.pixelSize: 10
                            }
                        }

                        Repeater {
                            model: modelsFolder

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.leftMargin: 12; Layout.rightMargin: 12
                                height: 56
                                radius: Theme.radius
                                color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 12

                                    Rectangle {
                                        width: 36; height: 36; radius: 18
                                        color: Qt.rgba(Theme.secondary.r, Theme.secondary.g, Theme.secondary.b, 0.15)
                                        Text { anchors.centerIn: parent; text: "\u2756"; font.pixelSize: 16; color: Theme.secondary }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: fileName; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                        Text { text: frameworkFromName(fileName) + " | " + formatSize(fileSize); color: Theme.textDim; font.pixelSize: 10 }
                                    }

                                    Components.StatusBadge { text: "Available"; badgeColor: Theme.success }

                                    Rectangle {
                                        width: 70; height: 28; radius: Theme.radiusSmall
                                        color: ldMa.containsMouse ? Theme.surfaceAlt : Theme.surface
                                        border.width: 1; border.color: Theme.primary

                                        Text { anchors.centerIn: parent; text: "Load"; color: Theme.primary; font.pixelSize: 10 }
                                        MouseArea {
                                            id: ldMa; anchors.fill: parent
                                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                            onClicked: NPIE.loadModel("/root/neuraos/models/" + fileName)
                                        }
                                    }
                                }
                            }
                        }

                        Item { height: 8 }
                    }

                    /* ─── NPU Tab ─── */
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        visible: activeTab === 2

                        Item { height: 12 }

                        Repeater {
                            model: NPUMonitor.devices

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.leftMargin: 12; Layout.rightMargin: 12
                                height: 72
                                radius: Theme.radius
                                color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 14; spacing: 14

                                    Rectangle {
                                        width: 44; height: 44; radius: 22
                                        color: Qt.rgba(Theme.secondary.r, Theme.secondary.g, Theme.secondary.b, 0.15)
                                        Text { anchors.centerIn: parent; text: "\u2756"; font.pixelSize: 20; color: Theme.secondary }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: modelData.name; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                        Text { text: modelData.type + " | " + modelData.cores + " cores | " + modelData.memoryMB + " MB"; color: Theme.textDim; font.pixelSize: 10 }
                                    }

                                    Column {
                                        spacing: 2
                                        Text { text: NPUMonitor.totalInferences + " inferences"; color: Theme.primary; font.pixelSize: 10 }
                                        Text { text: NPUMonitor.powerMw + " mW"; color: Theme.textDim; font.pixelSize: 10 }
                                    }
                                }
                            }
                        }

                        /* No NPU fallback */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            height: 60
                            radius: Theme.radius
                            color: Theme.surface
                            visible: NPUMonitor.deviceCount === 0

                            Text {
                                anchors.centerIn: parent
                                text: "No NPU devices detected"
                                color: Theme.textDim; font.pixelSize: 12
                            }
                        }

                        Item { height: 8 }
                    }

                    /* ─── Inference Tab ─── */
                    ColumnLayout {
                        width: parent.width
                        spacing: 10
                        visible: activeTab === 3

                        Item { height: 12 }

                        /* Inference history chart */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            height: 150
                            radius: Theme.radius
                            color: Theme.surface

                            Text {
                                x: 10; y: 10
                                text: "Inference Latency (ms)"
                                color: Theme.textDim; font.pixelSize: 11; font.bold: true
                            }

                            Canvas {
                                id: inferChart
                                anchors.fill: parent
                                anchors.topMargin: 28; anchors.margins: 10

                                property int tick: 0
                                property var latencyData: []

                                Timer {
                                    interval: 2000; running: true; repeat: true
                                    onTriggered: {
                                        inferChart.latencyData.push(NPIE.lastInferenceMs)
                                        if (inferChart.latencyData.length > 30) inferChart.latencyData.shift()
                                        inferChart.tick++
                                        inferChart.requestPaint()
                                    }
                                }

                                onPaint: {
                                    var ctx = getContext("2d")
                                    ctx.clearRect(0, 0, width, height)
                                    var data = latencyData
                                    if (data.length < 2) return

                                    var maxVal = 50
                                    for (var j = 0; j < data.length; j++) {
                                        if (data[j] > maxVal) maxVal = data[j] * 1.2
                                    }

                                    var step = width / 29
                                    ctx.beginPath()
                                    ctx.moveTo(0, height - (data[0] / maxVal * height))
                                    for (var i = 1; i < data.length; i++) {
                                        ctx.lineTo(i * step, height - (data[i] / maxVal * height))
                                    }
                                    ctx.strokeStyle = Theme.warning
                                    ctx.lineWidth = 2
                                    ctx.stroke()

                                    ctx.lineTo((data.length - 1) * step, height)
                                    ctx.lineTo(0, height)
                                    ctx.closePath()
                                    ctx.fillStyle = Qt.rgba(0.96, 0.62, 0.04, 0.08)
                                    ctx.fill()
                                }
                            }
                        }

                        /* Quick inference actions */
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            spacing: 10

                            Rectangle {
                                Layout.fillWidth: true; height: 44
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 10

                                    Text { text: "Total Inferences"; color: Theme.textDim; font.pixelSize: 11 }
                                    Item { Layout.fillWidth: true }
                                    Text { text: NPIE.inferenceCount.toString(); color: Theme.success; font.pixelSize: 14; font.bold: true }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 44
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 10

                                    Text { text: "Avg Latency"; color: Theme.textDim; font.pixelSize: 11 }
                                    Item { Layout.fillWidth: true }
                                    Text { text: NPIE.avgInferenceMs.toFixed(2) + " ms"; color: Theme.warning; font.pixelSize: 14; font.bold: true }
                                }
                            }
                        }

                        /* Run button */
                        Rectangle {
                            Layout.alignment: Qt.AlignHCenter
                            width: 160; height: 40
                            radius: Theme.radiusSmall
                            color: runInfMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary
                            visible: NPIE.modelLoaded

                            Text { anchors.centerIn: parent; text: "Run Inference"; color: "#000000"; font.bold: true; font.pixelSize: 12 }
                            MouseArea {
                                id: runInfMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: NPIE.runInference()
                            }
                        }

                        Item { height: 8 }
                    }
                }
            }
        }
    }
}
