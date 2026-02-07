import QtQuick 2.15
import QtQuick.Layouts 1.15
import Qt.labs.folderlistmodel 2.15
import ".."
import "../components" as Components

Item {
    id: fmApp
    anchors.fill: parent

    property string currentPath: "file:///root"
    property bool gridView: false

    FolderListModel {
        id: folderModel
        folder: currentPath
        showDirs: true
        showFiles: true
        showDirsFirst: true
        showDotAndDotDot: false
        sortField: FolderListModel.Name
    }

    function navigateTo(path) {
        currentPath = path
    }

    function navigateUp() {
        var parentFolder = folderModel.parentFolder
        if (parentFolder.toString() !== "") {
            currentPath = parentFolder
        }
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B"
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB"
        return (bytes / 1073741824).toFixed(1) + " GB"
    }

    function displayPath() {
        var s = currentPath.toString()
        if (s.indexOf("file://") === 0) s = s.substring(7)
        return s
    }

    function fileIcon(isDir, fileName) {
        if (isDir) return "folder"
        var ext = fileName.split('.').pop().toLowerCase()
        if (ext === "tflite" || ext === "onnx" || ext === "pt" || ext === "bin") return "neural"
        if (ext === "py" || ext === "sh" || ext === "js") return "code"
        if (ext === "log" || ext === "txt" || ext === "md") return "file"
        if (ext === "json" || ext === "yaml" || ext === "yml" || ext === "ini" || ext === "conf") return "gear"
        if (ext === "jpg" || ext === "png" || ext === "bmp") return "image"
        return "file"
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Sidebar + Content */
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* Sidebar */
                Rectangle {
                    Layout.fillHeight: true
                    width: 160
                    color: Theme.surface

                    Column {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 2

                        Text {
                            text: "Places"
                            color: Theme.textDim
                            font.pixelSize: 10
                            font.bold: true
                            topPadding: 4
                            bottomPadding: 8
                        }

                        Repeater {
                            model: ListModel {
                                ListElement { label: "Home"; path: "file:///root"; ico: "folder" }
                                ListElement { label: "Models"; path: "file:///root/neuraos/models"; ico: "neural" }
                                ListElement { label: "NeuralOS"; path: "file:///root/neuraos"; ico: "box" }
                                ListElement { label: "System"; path: "file:///"; ico: "gear" }
                                ListElement { label: "Tmp"; path: "file:///tmp"; ico: "file" }
                            }

                            Rectangle {
                                width: parent.width
                                height: 32
                                radius: Theme.radiusSmall
                                color: currentPath === model.path ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15) :
                                       sbMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 8
                                    spacing: 8

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: model.ico
                                        iconSize: 14
                                        iconColor: currentPath === model.path ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: model.label
                                        font.pixelSize: 12
                                        color: currentPath === model.path ? Theme.primary : Theme.text
                                        font.bold: currentPath === model.path
                                    }
                                }

                                MouseArea {
                                    id: sbMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo(model.path)
                                }
                            }
                        }

                        Item { height: 16 }

                        Text {
                            text: "Storage"
                            color: Theme.textDim
                            font.pixelSize: 10
                            font.bold: true
                            bottomPadding: 4
                        }

                        /* Disk usage bar */
                        Rectangle {
                            width: parent.width - 4
                            height: 6
                            radius: 3
                            color: Theme.surfaceLight

                            Rectangle {
                                width: parent.width * (SystemInfo.diskTotal > 0 ? SystemInfo.diskUsed / SystemInfo.diskTotal : 0)
                                height: parent.height
                                radius: 3
                                color: Theme.primary
                            }
                        }

                        Text {
                            text: (SystemInfo.diskUsed / 1073741824).toFixed(1) + " / " +
                                  (SystemInfo.diskTotal / 1073741824).toFixed(1) + " GB"
                            color: Theme.textDim
                            font.pixelSize: 10
                            topPadding: 2
                        }
                    }
                }

                /* Separator */
                Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

                /* Content area */
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 0

                    /* Toolbar */
                    Rectangle {
                        Layout.fillWidth: true
                        height: 40
                        color: Theme.surface

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 8
                            anchors.rightMargin: 8
                            spacing: 6

                            /* Back */
                            Rectangle {
                                width: 30; height: 30
                                radius: Theme.radiusSmall
                                color: backMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Components.CanvasIcon { anchors.centerIn: parent; iconName: "arrow-left"; iconColor: Theme.text; iconSize: 16 }
                                MouseArea {
                                    id: backMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateUp()
                                }
                            }

                            /* Path */
                            Rectangle {
                                Layout.fillWidth: true
                                height: 28
                                radius: Theme.radiusSmall
                                color: Theme.surfaceAlt

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.left: parent.left
                                    anchors.leftMargin: 10
                                    anchors.right: parent.right
                                    anchors.rightMargin: 10
                                    text: displayPath()
                                    color: Theme.text
                                    font.pixelSize: 12
                                    font.family: "monospace"
                                    elide: Text.ElideMiddle
                                }
                            }

                            /* Grid/List toggle */
                            Rectangle {
                                width: 30; height: 30
                                radius: Theme.radiusSmall
                                color: viewMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: gridView ? "list" : "grid"
                                    iconColor: Theme.primary; iconSize: 16
                                }
                                MouseArea {
                                    id: viewMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: gridView = !gridView
                                }
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                    /* File list or grid */
                    Flickable {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentHeight: gridView ? gridContent.height : listContent.height
                        clip: true

                        /* List View */
                        Column {
                            id: listContent
                            width: parent.width
                            visible: !gridView

                            /* Column header */
                            Row {
                                width: parent.width
                                height: 30
                                spacing: 0

                                Item { width: 8; height: 1 }
                                Text { width: 36;  text: ""; height: 30; verticalAlignment: Text.AlignVCenter; color: Theme.textDim; font.pixelSize: 10 }
                                Text { width: parent.width * 0.4; text: "Name"; height: 30; verticalAlignment: Text.AlignVCenter; color: Theme.textDim; font.pixelSize: 10; font.bold: true }
                                Text { width: 80; text: "Size"; height: 30; verticalAlignment: Text.AlignVCenter; color: Theme.textDim; font.pixelSize: 10; font.bold: true }
                                Text { width: 120; text: "Modified"; height: 30; verticalAlignment: Text.AlignVCenter; color: Theme.textDim; font.pixelSize: 10; font.bold: true }
                            }

                            Rectangle { width: parent.width; height: 1; color: Theme.surfaceLight }

                            Repeater {
                                model: folderModel

                                Rectangle {
                                    width: parent.width
                                    height: 34
                                    color: flMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                    Row {
                                        anchors.fill: parent
                                        anchors.leftMargin: 8
                                        spacing: 0

                                        Item { width: 36; height: 16; anchors.verticalCenter: parent.verticalCenter; Components.CanvasIcon { anchors.centerIn: parent; iconName: fileIcon(fileIsDir, fileName); iconSize: 16; iconColor: fileIsDir ? Theme.primary : Theme.textDim } }
                                        Text {
                                            width: parent.parent.width * 0.4
                                            text: fileName
                                            color: fileIsDir ? Theme.primary : Theme.text
                                            font.pixelSize: 12; font.bold: fileIsDir
                                            elide: Text.ElideRight
                                            anchors.verticalCenter: parent.verticalCenter
                                        }
                                        Text { width: 80; text: fileIsDir ? "--" : formatSize(fileSize); color: Theme.textDim; font.pixelSize: 11; anchors.verticalCenter: parent.verticalCenter }
                                        Text { width: 120; text: Qt.formatDateTime(fileModified, "yyyy-MM-dd HH:mm"); color: Theme.textDim; font.pixelSize: 11; anchors.verticalCenter: parent.verticalCenter }
                                    }

                                    MouseArea {
                                        id: flMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onDoubleClicked: {
                                            if (fileIsDir) {
                                                navigateTo(fileURL)
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /* Grid View */
                        Flow {
                            id: gridContent
                            width: parent.width
                            padding: 12
                            spacing: 10
                            visible: gridView

                            Repeater {
                                model: folderModel

                                Rectangle {
                                    width: 86; height: 90
                                    radius: Theme.radiusSmall
                                    color: grMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                    Column {
                                        anchors.centerIn: parent
                                        spacing: 4

                                        Components.CanvasIcon {
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            iconName: fileIcon(fileIsDir, fileName)
                                            iconSize: 28
                                            iconColor: fileIsDir ? Theme.primary : Theme.textDim
                                        }
                                        Text {
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            width: 76
                                            text: fileName
                                            color: fileIsDir ? Theme.primary : Theme.text
                                            font.pixelSize: 10
                                            horizontalAlignment: Text.AlignHCenter
                                            elide: Text.ElideMiddle
                                        }
                                    }

                                    MouseArea {
                                        id: grMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onDoubleClicked: {
                                            if (fileIsDir) {
                                                navigateTo(fileURL)
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
                        height: 26
                        color: Theme.surface

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 10
                            anchors.rightMargin: 10

                            Text { text: folderModel.count + " items"; color: Theme.textDim; font.pixelSize: 10 }
                            Item { Layout.fillWidth: true }
                            Text { text: displayPath(); color: Theme.textDim; font.pixelSize: 10; font.family: "monospace" }
                        }
                    }
                }
            }
        }
    }
}
