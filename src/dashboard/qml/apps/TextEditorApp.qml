import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: editorApp
    anchors.fill: parent

    property int currentTab: 0
    property var tabs: [
        { title: "untitled.txt", content: "Welcome to NeuralOS Text Editor.\n\nStart typing here...\n", modified: false },
        { title: "config.yaml", content: "# NeuralOS Configuration\nversion: 3.3.0\n\nsystem:\n  hostname: neuralbox\n  timezone: UTC\n  locale: en_US.UTF-8\n\nai_runtime:\n  backend: cuda\n  precision: fp16\n  max_memory: 8192\n  models_path: /opt/neural/models\n\nnetwork:\n  interface: eth0\n  dns:\n    - 1.1.1.1\n    - 8.8.8.8\n  firewall: enabled\n\nservices:\n  ssh: enabled\n  http: disabled\n  npu_daemon: enabled\n", modified: false }
    ]

    function getCursorInfo() {
        var text = textEdit.text.substring(0, textEdit.cursorPosition)
        var lines = text.split("\n")
        var ln = lines.length
        var col = lines[lines.length - 1].length + 1
        return "Ln " + ln + ", Col " + col
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Toolbar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 36
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 2

                    Repeater {
                        model: [
                            { icon: "file", tip: "New" },
                            { icon: "save", tip: "Save" },
                            { icon: "separator" },
                            { icon: "arrow-left", tip: "Undo" },
                            { icon: "arrow-right", tip: "Redo" },
                            { icon: "separator" },
                            { icon: "copy", tip: "Copy" },
                            { icon: "edit", tip: "Paste" },
                            { icon: "search", tip: "Find" }
                        ]

                        Rectangle {
                            Layout.preferredWidth: modelData.icon === "separator" ? 1 : 30
                            Layout.preferredHeight: modelData.icon === "separator" ? 20 : 28
                            Layout.alignment: Qt.AlignVCenter
                            radius: modelData.icon === "separator" ? 0 : Theme.radiusSmall
                            color: modelData.icon === "separator" ? Theme.surfaceLight
                                 : tbBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Components.CanvasIcon {
                                visible: modelData.icon !== "separator"
                                anchors.centerIn: parent
                                iconName: modelData.icon || ""
                                iconSize: 14
                                iconColor: Theme.textDim
                            }

                            MouseArea {
                                id: tbBtnMa
                                anchors.fill: parent
                                hoverEnabled: true
                                enabled: modelData.icon !== "separator"
                                cursorShape: modelData.icon !== "separator" ? Qt.PointingHandCursor : Qt.ArrowCursor
                            }

                            ToolTip {
                                visible: tbBtnMa.containsMouse && modelData.tip
                                text: modelData.tip || ""
                                delay: 500
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Tab Bar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 32
                color: Theme.surfaceAlt

                Row {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.leftMargin: 4
                    spacing: 1

                    Repeater {
                        model: tabs.length

                        Rectangle {
                            width: tabLabel.implicitWidth + 40
                            height: 28
                            radius: Theme.radiusSmall
                            color: index === currentTab ? Theme.surface : "transparent"

                            Row {
                                anchors.centerIn: parent
                                spacing: 6

                                Text {
                                    id: tabLabel
                                    text: tabs[index].title + (tabs[index].modified ? " •" : "")
                                    font.pixelSize: 12
                                    color: index === currentTab ? Theme.text : Theme.textDim
                                }

                                Rectangle {
                                    width: 16; height: 16
                                    radius: 8
                                    color: tabCloseMa.containsMouse ? Theme.surfaceLight : "transparent"
                                    anchors.verticalCenter: parent.verticalCenter

                                    Text {
                                        anchors.centerIn: parent
                                        text: "×"
                                        font.pixelSize: 12
                                        color: Theme.textDim
                                    }

                                    MouseArea {
                                        id: tabCloseMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                z: -1
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    currentTab = index
                                    textEdit.text = tabs[currentTab].content
                                }
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Editor Area */
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: Theme.background

                RowLayout {
                    anchors.fill: parent
                    spacing: 0

                    /* Line Numbers */
                    Rectangle {
                        Layout.preferredWidth: 48
                        Layout.fillHeight: true
                        color: Theme.surfaceAlt

                        Column {
                            anchors.top: parent.top
                            anchors.topMargin: 10 - editorFlick.contentY % 1000000
                            anchors.right: parent.right
                            anchors.rightMargin: 8
                            spacing: 0

                            Repeater {
                                model: Math.max(textEdit.text.split("\n").length, 1)

                                Text {
                                    text: (index + 1).toString()
                                    font.family: "monospace"
                                    font.pixelSize: 13
                                    color: Theme.textMuted
                                    height: Math.ceil(13 * 1.45)
                                    horizontalAlignment: Text.AlignRight
                                    width: 32
                                }
                            }
                        }
                    }

                    Rectangle {
                        Layout.preferredWidth: 1
                        Layout.fillHeight: true
                        color: Theme.surfaceLight
                    }

                    /* Text Content */
                    Flickable {
                        id: editorFlick
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentWidth: textEdit.width
                        contentHeight: textEdit.height
                        clip: true
                        flickableDirection: Flickable.VerticalFlick

                        TextEdit {
                            id: textEdit
                            width: editorFlick.width
                            padding: 10
                            font.family: "monospace"
                            font.pixelSize: 13
                            color: Theme.text
                            selectionColor: Theme.primary
                            selectedTextColor: "#FFFFFF"
                            wrapMode: TextEdit.WrapAtWordBoundaryOrAnywhere
                            text: tabs[currentTab].content
                            onTextChanged: {
                                if (tabs[currentTab]) {
                                    var t = tabs.slice()
                                    t[currentTab] = Object.assign({}, t[currentTab], { content: text, modified: true })
                                    tabs = t
                                }
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Status Bar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 24
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12
                    spacing: 16

                    Text {
                        text: getCursorInfo()
                        font.pixelSize: 11
                        color: Theme.textDim
                    }

                    Rectangle { width: 1; height: 12; color: Theme.surfaceLight }

                    Text {
                        text: "UTF-8"
                        font.pixelSize: 11
                        color: Theme.textDim
                    }

                    Rectangle { width: 1; height: 12; color: Theme.surfaceLight }

                    Text {
                        text: tabs[currentTab].title.indexOf(".yaml") >= 0 ? "YAML" : "Plain Text"
                        font.pixelSize: 11
                        color: Theme.textDim
                    }

                    Item { Layout.fillWidth: true }

                    Text {
                        text: textEdit.text.length + " characters"
                        font.pixelSize: 11
                        color: Theme.textDim
                    }
                }
            }
        }
    }
}
