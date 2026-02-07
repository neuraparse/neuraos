import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."

Rectangle {
    id: cmdPalette
    width: Theme.commandPaletteW
    height: Theme.commandPaletteH
    radius: Theme.radius
    color: Theme.commandPaletteBg
    border.width: 1
    border.color: Theme.glassBorder

    signal commandSelected(string action, string target)
    signal closed()

    /* ─── Top inner glow ─── */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 1; anchors.leftMargin: 1; anchors.rightMargin: 1
        height: 1; radius: Theme.radius
        color: Qt.rgba(1, 1, 1, 0.05)
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        /* ─── Search Input ─── */
        Rectangle {
            Layout.fillWidth: true
            height: 48
            radius: Theme.radiusSmall
            color: Theme.surfaceAlt
            border.width: 1
            border.color: cmdInput.activeFocus ? Theme.primary : Theme.glassBorder

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10

                /* Search icon */
                Text {
                    text: "\u2315"
                    color: Theme.textDim
                    font.pixelSize: 18
                }

                TextInput {
                    id: cmdInput
                    Layout.fillWidth: true
                    color: Theme.text
                    font.pixelSize: Theme.fontSizeLarge
                    font.family: Theme.fontFamily
                    clip: true
                    focus: true
                    selectByMouse: true

                    property string placeholderText: "Type a command... (e.g. 'open terminal', 'system status')"

                    Text {
                        anchors.fill: parent
                        anchors.verticalCenter: parent.verticalCenter
                        text: cmdInput.placeholderText
                        color: Theme.textMuted
                        font.pixelSize: Theme.fontSizeLarge
                        visible: !cmdInput.text && !cmdInput.activeFocus
                    }

                    onTextChanged: {
                        var results = CommandEngine.getSuggestions(text)
                        suggestionModel.clear()
                        for (var i = 0; i < results.length; i++) {
                            suggestionModel.append(results[i])
                        }
                    }

                    Keys.onReturnPressed: {
                        if (cmdInput.text.length > 0) {
                            var result = CommandEngine.execute(cmdInput.text)
                            if (result.matched) {
                                commandSelected(result.action, result.target || "")
                            }
                            cmdInput.text = ""
                            closed()
                        }
                    }

                    Keys.onEscapePressed: {
                        cmdInput.text = ""
                        closed()
                    }

                    Keys.onDownPressed: {
                        suggestionList.incrementCurrentIndex()
                    }

                    Keys.onUpPressed: {
                        suggestionList.decrementCurrentIndex()
                    }
                }

                /* Clear button */
                Rectangle {
                    width: 24; height: 24
                    radius: 12
                    color: clearMa.containsMouse ? Theme.surfaceLight : "transparent"
                    visible: cmdInput.text.length > 0

                    Text {
                        anchors.centerIn: parent
                        text: "\u2715"
                        color: Theme.textDim
                        font.pixelSize: 12
                    }

                    MouseArea {
                        id: clearMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: cmdInput.text = ""
                    }
                }
            }
        }

        /* ─── Divider ─── */
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: Theme.glassBorder
        }

        /* ─── Suggestions List ─── */
        ListModel {
            id: suggestionModel
        }

        ListView {
            id: suggestionList
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: suggestionModel
            clip: true
            spacing: 2
            currentIndex: -1

            delegate: Rectangle {
                width: suggestionList.width
                height: 44
                radius: Theme.radiusTiny
                color: {
                    if (suggestionList.currentIndex === index) return Theme.glassActive
                    if (suggItemMa.containsMouse) return Theme.glassHover
                    return "transparent"
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12; anchors.rightMargin: 12
                    spacing: 12

                    /* Icon indicator */
                    Rectangle {
                        width: 28; height: 28
                        radius: Theme.radiusTiny
                        color: {
                            var t = model.type || "command"
                            if (t === "history") return Qt.rgba(Theme.textDim.r, Theme.textDim.g, Theme.textDim.b, 0.15)
                            return Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                        }

                        Text {
                            anchors.centerIn: parent
                            text: {
                                var icon = model.icon || "terminal"
                                if (icon === "terminal") return "\u2318"
                                if (icon === "folder") return "\u{1F4C1}"
                                if (icon === "gear") return "\u2699"
                                if (icon === "monitor") return "\u{1F5A5}"
                                if (icon === "robot") return "\u2604"
                                if (icon === "neural") return "\u2B22"
                                if (icon === "hub") return "\u2B21"
                                if (icon === "brain") return "\u25C8"
                                if (icon === "automation") return "\u21BB"
                                if (icon === "plug") return "\u26A1"
                                if (icon === "book") return "\u{1F4D6}"
                                if (icon === "network") return "\u2B2A"
                                if (icon === "chip") return "\u2B23"
                                if (icon === "wifi") return "\u2022"
                                if (icon === "clock") return "\u23F0"
                                if (icon === "volume") return "\u266B"
                                if (icon === "globe") return "\u25CE"
                                if (icon === "edit") return "\u270E"
                                if (icon === "grid") return "\u2593"
                                return "\u25B6"
                            }
                            color: {
                                var t = model.type || "command"
                                if (t === "history") return Theme.textDim
                                return Theme.primary
                            }
                            font.pixelSize: 13
                        }
                    }

                    /* Command text */
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 1

                        Text {
                            text: model.text || ""
                            color: Theme.text
                            font.pixelSize: 13
                            font.bold: (model.type || "") !== "history"
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                        Text {
                            text: model.description || ""
                            color: Theme.textDim
                            font.pixelSize: 11
                            elide: Text.ElideRight
                            visible: (model.description || "") !== ""
                            Layout.fillWidth: true
                        }
                    }

                    /* Type badge */
                    Rectangle {
                        width: typeBadge.implicitWidth + 12
                        height: 20
                        radius: 10
                        color: Qt.rgba(Theme.textDim.r, Theme.textDim.g, Theme.textDim.b, 0.12)

                        Text {
                            id: typeBadge
                            anchors.centerIn: parent
                            text: (model.type || "command").toUpperCase()
                            color: Theme.textDim
                            font.pixelSize: 9
                            font.bold: true
                        }
                    }
                }

                MouseArea {
                    id: suggItemMa
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        var text = model.text || ""
                        if (model.type === "history") {
                            cmdInput.text = text
                        } else {
                            var result = CommandEngine.execute(text)
                            if (result.matched) {
                                commandSelected(result.action, result.target || "")
                            }
                            cmdInput.text = ""
                            closed()
                        }
                    }
                }
            }

            /* Empty state */
            Text {
                anchors.centerIn: parent
                text: cmdInput.text.length > 0 ? "No matching commands" : "Start typing to search commands..."
                color: Theme.textMuted
                font.pixelSize: 13
                visible: suggestionModel.count === 0
            }
        }

        /* ─── Footer ─── */
        Rectangle {
            Layout.fillWidth: true
            height: 28

            color: "transparent"

            RowLayout {
                anchors.fill: parent
                spacing: 16

                Text {
                    text: "\u2191\u2193 Navigate"
                    color: Theme.textMuted
                    font.pixelSize: 10
                }
                Text {
                    text: "\u23CE Execute"
                    color: Theme.textMuted
                    font.pixelSize: 10
                }
                Text {
                    text: "Esc Close"
                    color: Theme.textMuted
                    font.pixelSize: 10
                }
                Item { Layout.fillWidth: true }
                Text {
                    text: "Ctrl+K"
                    color: Theme.textMuted
                    font.pixelSize: 10
                    font.bold: true
                }
            }
        }
    }

    Component.onCompleted: {
        /* Load initial suggestions (history) */
        var results = CommandEngine.getSuggestions("")
        for (var i = 0; i < results.length; i++) {
            suggestionModel.append(results[i])
        }
        cmdInput.forceActiveFocus()
    }
}
