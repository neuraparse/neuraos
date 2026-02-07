import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: filePage

    property string currentPath: "/home"
    property var fileList: []
    property bool gridView: false

    Component.onCompleted: loadDirectory(currentPath)

    function loadDirectory(path) {
        currentPath = path
        /* Simulated file system for demo */
        if (path === "/home") {
            fileList = [
                { name: "models", isDir: true, size: "", modified: "2025-01-15", perms: "drwxr-xr-x" },
                { name: "data", isDir: true, size: "", modified: "2025-01-14", perms: "drwxr-xr-x" },
                { name: "logs", isDir: true, size: "", modified: "2025-01-15", perms: "drwxr-xr-x" },
                { name: "scripts", isDir: true, size: "", modified: "2025-01-10", perms: "drwxr-xr-x" },
                { name: "config.ini", isDir: false, size: "2.4 KB", modified: "2025-01-15", perms: "-rw-r--r--" },
                { name: "README.md", isDir: false, size: "8.1 KB", modified: "2025-01-12", perms: "-rw-r--r--" },
                { name: "startup.sh", isDir: false, size: "1.2 KB", modified: "2025-01-09", perms: "-rwxr-xr-x" },
                { name: "system.log", isDir: false, size: "45.3 KB", modified: "2025-01-15", perms: "-rw-r--r--" }
            ]
        } else if (path === "/home/models") {
            fileList = [
                { name: "..", isDir: true, size: "", modified: "", perms: "" },
                { name: "demo_model.tflite", isDir: false, size: "4.2 MB", modified: "2025-01-15", perms: "-rw-r--r--" },
                { name: "classifier.onnx", isDir: false, size: "12.8 MB", modified: "2025-01-14", perms: "-rw-r--r--" },
                { name: "embeddings.bin", isDir: false, size: "128 MB", modified: "2025-01-10", perms: "-rw-r--r--" }
            ]
        } else if (path === "/home/data") {
            fileList = [
                { name: "..", isDir: true, size: "", modified: "", perms: "" },
                { name: "training_set.csv", isDir: false, size: "2.1 MB", modified: "2025-01-14", perms: "-rw-r--r--" },
                { name: "labels.json", isDir: false, size: "156 KB", modified: "2025-01-14", perms: "-rw-r--r--" },
                { name: "benchmark_results.json", isDir: false, size: "24 KB", modified: "2025-01-15", perms: "-rw-r--r--" }
            ]
        } else {
            fileList = [
                { name: "..", isDir: true, size: "", modified: "", perms: "" }
            ]
        }
        fileListChanged()
    }

    function navigateUp() {
        var parts = currentPath.split("/")
        if (parts.length > 2) {
            parts.pop()
            loadDirectory(parts.join("/"))
        }
    }

    Flickable {
        anchors.fill: parent
        anchors.margins: 20
        contentHeight: col.height
        clip: true

        ColumnLayout {
            id: col
            width: parent.width
            spacing: 12

            Components.SectionHeader { title: "File Manager" }

            /* Breadcrumb + Controls */
            Rectangle {
                Layout.fillWidth: true
                height: 44
                radius: Theme.radius
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12
                    spacing: 8

                    /* Back button */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: backMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Text {
                            anchors.centerIn: parent
                            text: "\u2190"
                            color: Theme.text
                            font.pixelSize: 16
                        }

                        MouseArea {
                            id: backMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: navigateUp()
                        }
                    }

                    /* Breadcrumb path */
                    Text {
                        Layout.fillWidth: true
                        text: currentPath
                        color: Theme.text
                        font.pixelSize: Theme.fontSizeNormal
                        font.family: "Liberation Mono"
                        elide: Text.ElideMiddle
                    }

                    /* Grid/List toggle */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: viewMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Text {
                            anchors.centerIn: parent
                            text: gridView ? "\u2261" : "\u2637"
                            color: Theme.primary
                            font.pixelSize: 16
                        }

                        MouseArea {
                            id: viewMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: gridView = !gridView
                        }
                    }
                }
            }

            /* File list view */
            Rectangle {
                Layout.fillWidth: true
                height: Math.max(400, fileList.length * 40 + 44)
                radius: Theme.radius
                color: Theme.surface
                clip: true
                visible: !gridView

                Column {
                    anchors.fill: parent
                    anchors.margins: 8

                    /* Header */
                    Row {
                        width: parent.width
                        height: 36
                        spacing: 0

                        Text { width: 40;  text: ""; color: Theme.textDim; font.pixelSize: 11 }
                        Text { width: parent.width * 0.35; text: "Name"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 100; text: "Size"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 120; text: "Modified"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                        Text { width: 110; text: "Permissions"; color: Theme.textDim; font.pixelSize: 11; font.bold: true }
                    }

                    /* Separator */
                    Rectangle {
                        width: parent.width
                        height: 1
                        color: Theme.surfaceLight
                    }

                    /* File entries */
                    Repeater {
                        model: fileList.length

                        Rectangle {
                            width: parent.width
                            height: 38
                            color: fileMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            radius: Theme.radiusSmall

                            Row {
                                anchors.fill: parent
                                anchors.leftMargin: 4
                                spacing: 0

                                /* Icon */
                                Text {
                                    width: 40
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: fileList[index].isDir ? "\uD83D\uDCC1" : "\uD83D\uDCC4"
                                    font.pixelSize: 16
                                    horizontalAlignment: Text.AlignHCenter
                                }

                                /* Name */
                                Text {
                                    width: parent.parent.width * 0.35
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: fileList[index].name
                                    color: fileList[index].isDir ? Theme.primary : Theme.text
                                    font.pixelSize: 12
                                    font.bold: fileList[index].isDir
                                    elide: Text.ElideRight
                                }

                                /* Size */
                                Text {
                                    width: 100
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: fileList[index].size
                                    color: Theme.textDim
                                    font.pixelSize: 11
                                }

                                /* Modified */
                                Text {
                                    width: 120
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: fileList[index].modified
                                    color: Theme.textDim
                                    font.pixelSize: 11
                                }

                                /* Permissions */
                                Text {
                                    width: 110
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: fileList[index].perms
                                    color: Theme.textDim
                                    font.pixelSize: 11
                                    font.family: "Liberation Mono"
                                }
                            }

                            MouseArea {
                                id: fileMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onDoubleClicked: {
                                    if (fileList[index].isDir) {
                                        if (fileList[index].name === "..") {
                                            navigateUp()
                                        } else {
                                            loadDirectory(currentPath + "/" + fileList[index].name)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Grid view */
            Rectangle {
                Layout.fillWidth: true
                height: Math.max(400, Math.ceil(fileList.length / 5) * 110 + 20)
                radius: Theme.radius
                color: Theme.surface
                clip: true
                visible: gridView

                Flow {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 12

                    Repeater {
                        model: fileList.length

                        Rectangle {
                            width: 90; height: 100
                            radius: Theme.radiusSmall
                            color: gridMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Column {
                                anchors.centerIn: parent
                                spacing: 6

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: fileList[index].isDir ? "\uD83D\uDCC1" : "\uD83D\uDCC4"
                                    font.pixelSize: 32
                                }

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    width: 80
                                    text: fileList[index].name
                                    color: fileList[index].isDir ? Theme.primary : Theme.text
                                    font.pixelSize: 10
                                    horizontalAlignment: Text.AlignHCenter
                                    elide: Text.ElideMiddle
                                }
                            }

                            MouseArea {
                                id: gridMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onDoubleClicked: {
                                    if (fileList[index].isDir) {
                                        if (fileList[index].name === "..") {
                                            navigateUp()
                                        } else {
                                            loadDirectory(currentPath + "/" + fileList[index].name)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Status bar */
            Rectangle {
                Layout.fillWidth: true
                height: 32
                radius: Theme.radiusSmall
                color: Theme.surfaceAlt

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12

                    Text {
                        text: fileList.length + " items"
                        color: Theme.textDim
                        font.pixelSize: Theme.fontSizeSmall
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: currentPath
                        color: Theme.textDim
                        font.pixelSize: Theme.fontSizeSmall
                        font.family: "Liberation Mono"
                    }
                }
            }

            Item { height: 20 }
        }
    }
}
