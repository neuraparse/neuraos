import QtQuick 2.15
import QtQuick.Layouts 1.15
import Qt.labs.folderlistmodel 2.15
import ".."
import "../components" as Components

Item {
    id: pkgApp
    anchors.fill: parent

    property int activeTab: 0
    property string searchText: ""

    FolderListModel {
        id: modelsFolder
        folder: "file:///root/neuraos/models"
        nameFilters: ["*.tflite", "*.onnx", "*.bin", "*.pt", "*.so", "*.cfg"]
        showDirs: false
        sortField: FolderListModel.Name
    }

    ListModel {
        id: availableModel
        ListElement { name: "yolov8n.onnx"; category: "AI Model"; version: "8.0.0"; size: "6.3 MB"; desc: "YOLOv8 nano object detection" }
        ListElement { name: "whisper-tiny"; category: "AI Model"; version: "1.0.0"; size: "39 MB"; desc: "Speech-to-text model" }
        ListElement { name: "llama-3-1B"; category: "AI Model"; version: "3.0.0"; size: "1.1 GB"; desc: "Small language model for edge" }
        ListElement { name: "stable-diffusion-nano"; category: "AI Model"; version: "1.5.0"; size: "512 MB"; desc: "Image generation for edge devices" }
        ListElement { name: "tflite-runtime"; category: "Runtime"; version: "2.16.0"; size: "18 MB"; desc: "TensorFlow Lite inference runtime" }
        ListElement { name: "onnxruntime-arm"; category: "Runtime"; version: "1.17.0"; size: "22 MB"; desc: "ONNX Runtime for ARM64" }
        ListElement { name: "ffmpeg-lite"; category: "Tool"; version: "6.1.0"; size: "15 MB"; desc: "Lightweight media processing" }
        ListElement { name: "mosquitto"; category: "Service"; version: "2.0.18"; size: "1.2 MB"; desc: "MQTT broker for IoT" }
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B"
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB"
        return (bytes / 1073741824).toFixed(1) + " GB"
    }

    function categoryFromName(name) {
        var ext = name.split('.').pop().toLowerCase()
        if (ext === "tflite" || ext === "onnx" || ext === "pt") return "AI Model"
        if (ext === "so") return "Library"
        if (ext === "bin") return "Binary"
        if (ext === "cfg") return "Config"
        return "File"
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Top bar */
            Rectangle {
                Layout.fillWidth: true; height: 42
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 8; spacing: 8

                    /* Tabs */
                    Repeater {
                        model: ["Installed", "Available", "Updates"]

                        Rectangle {
                            width: 80; height: 28; radius: Theme.radiusSmall
                            color: activeTab === index ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2) :
                                   ptMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent; text: modelData
                                color: activeTab === index ? Theme.primary : Theme.text
                                font.pixelSize: 11; font.bold: activeTab === index
                            }
                            MouseArea { id: ptMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: activeTab = index }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Search */
                    Rectangle {
                        width: 180; height: 28
                        radius: Theme.radiusSmall
                        color: Theme.surfaceAlt

                        TextInput {
                            anchors.fill: parent; anchors.margins: 6
                            color: Theme.text; font.pixelSize: 11
                            clip: true; selectByMouse: true
                            onTextChanged: searchText = text

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Search packages..."
                                color: Theme.textMuted; font.pixelSize: 11
                                visible: !parent.text && !parent.activeFocus
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Content */
            Flickable {
                Layout.fillWidth: true; Layout.fillHeight: true
                contentHeight: pkgContent.height; clip: true

                Column {
                    id: pkgContent
                    width: parent.width

                    /* Installed - real model files */
                    ColumnLayout {
                        width: parent.width; spacing: 2; visible: activeTab === 0

                        /* Column headers */
                        Rectangle {
                            Layout.fillWidth: true; height: 32; color: Theme.surface

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 12; spacing: 10

                                Text { text: "Package"; color: Theme.textDim; font.pixelSize: 10; font.bold: true; Layout.fillWidth: true }
                                Text { text: "Category"; color: Theme.textDim; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 80 }
                                Text { text: "Size"; color: Theme.textDim; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 80 }
                                Item { Layout.preferredWidth: 70 }
                            }
                        }

                        /* Info bar */
                        Rectangle {
                            Layout.fillWidth: true; height: 28; color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.05)

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.left: parent.left; anchors.leftMargin: 12
                                text: modelsFolder.count + " files in /root/neuraos/models"
                                color: Theme.textDim; font.pixelSize: 10
                            }
                        }

                        Repeater {
                            model: modelsFolder

                            Rectangle {
                                Layout.fillWidth: true; height: 44
                                color: ipkMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                                    Column {
                                        Layout.fillWidth: true; spacing: 1
                                        Text { text: fileName; color: Theme.text; font.pixelSize: 11; font.bold: true }
                                    }

                                    Text { text: categoryFromName(fileName); color: Theme.textDim; font.pixelSize: 10; Layout.preferredWidth: 80 }
                                    Text { text: formatSize(fileSize); color: Theme.textDim; font.pixelSize: 10; Layout.preferredWidth: 80 }

                                    Rectangle {
                                        width: 65; height: 24; radius: Theme.radiusSmall; Layout.preferredWidth: 70
                                        color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.1)
                                        Text { anchors.centerIn: parent; text: "Load"; color: Theme.primary; font.pixelSize: 9 }
                                        MouseArea {
                                            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                            onClicked: NPIE.loadModel("/root/neuraos/models/" + fileName)
                                        }
                                    }
                                }

                                MouseArea { id: ipkMa; anchors.fill: parent; hoverEnabled: true; z: -1 }
                            }
                        }
                    }

                    /* Available */
                    ColumnLayout {
                        width: parent.width; spacing: 2; visible: activeTab === 1

                        Repeater {
                            model: availableModel

                            Rectangle {
                                Layout.fillWidth: true; height: 60
                                color: apkMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                                    Rectangle {
                                        width: 36; height: 36; radius: 18
                                        color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.1)
                                        Text { anchors.centerIn: parent; text: "\u2630"; font.pixelSize: 16; color: Theme.primary }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: model.name; color: Theme.text; font.pixelSize: 11; font.bold: true }
                                        Text { text: model.desc; color: Theme.textDim; font.pixelSize: 9; elide: Text.ElideRight; width: 280 }
                                    }

                                    Column {
                                        spacing: 2
                                        Text { text: model.version; color: Theme.textDim; font.pixelSize: 9; font.family: "monospace" }
                                        Text { text: model.size; color: Theme.textDim; font.pixelSize: 9 }
                                    }

                                    Rectangle {
                                        width: 65; height: 26; radius: Theme.radiusSmall
                                        color: instMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary
                                        Text { anchors.centerIn: parent; text: "Install"; color: "#000"; font.bold: true; font.pixelSize: 10 }
                                        MouseArea { id: instMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                                    }
                                }

                                MouseArea { id: apkMa; anchors.fill: parent; hoverEnabled: true; z: -1 }
                            }
                        }
                    }

                    /* Updates */
                    ColumnLayout {
                        width: parent.width; spacing: 8; visible: activeTab === 2

                        Item { height: 10 }

                        Rectangle {
                            Layout.fillWidth: true; Layout.margins: 10; height: 80
                            radius: Theme.radius; color: Theme.surface

                            Column {
                                anchors.centerIn: parent; spacing: 6
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "\u2714"; font.pixelSize: 28; color: Theme.success }
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "All packages are up to date"; color: Theme.text; font.pixelSize: 12 }
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Last checked: just now"; color: Theme.textDim; font.pixelSize: 10 }
                            }
                        }

                        Rectangle {
                            Layout.alignment: Qt.AlignHCenter
                            width: 120; height: 32; radius: Theme.radiusSmall
                            color: chkMa.containsMouse ? Theme.surfaceAlt : Theme.surface
                            border.width: 1; border.color: Theme.primary
                            Text { anchors.centerIn: parent; text: "Check Updates"; color: Theme.primary; font.pixelSize: 10 }
                            MouseArea { id: chkMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                        }

                        Item { height: 8 }
                    }
                }
            }
        }
    }
}
