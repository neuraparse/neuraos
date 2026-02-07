import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: kbApp
    anchors.fill: parent

    property int activeTab: 0
    property string searchQuery: ""
    property var searchResults: []
    property string qaAnswer: ""

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Top Bar ─── */
            Rectangle {
                Layout.fillWidth: true; height: 46; color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 10; spacing: 12

                    Text { text: "Knowledge Base"; color: Theme.knowledge; font.pixelSize: 15; font.bold: true }

                    Item { width: 12 }

                    Repeater {
                        model: ["Documents", "Search", "Ask AI"]

                        Rectangle {
                            width: 80; height: 30; radius: Theme.radiusSmall
                            color: activeTab === index ?
                                Qt.rgba(Theme.knowledge.r, Theme.knowledge.g, Theme.knowledge.b, 0.2) :
                                kbTabMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Text {
                                anchors.centerIn: parent; text: modelData
                                color: activeTab === index ? Theme.knowledge : Theme.text
                                font.pixelSize: 11; font.bold: activeTab === index
                            }
                            MouseArea {
                                id: kbTabMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: activeTab = index
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Index status */
                    Rectangle {
                        width: idxRow.width + 16; height: 26; radius: 13
                        color: Knowledge.indexStatus === "ready" ?
                            Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15) :
                            Qt.rgba(Theme.warning.r, Theme.warning.g, Theme.warning.b, 0.15)
                        Row {
                            id: idxRow; anchors.centerIn: parent; spacing: 6
                            Rectangle { width: 8; height: 8; radius: 4; color: Knowledge.indexStatus === "ready" ? Theme.success : Theme.warning }
                            Text {
                                text: Knowledge.indexStatus === "ready" ? "Index Ready" : "Indexing..."
                                color: Knowledge.indexStatus === "ready" ? Theme.success : Theme.warning
                                font.pixelSize: 10; font.bold: true
                            }
                        }
                    }

                    Row {
                        spacing: 12
                        StatPill { label: "Docs"; value: Knowledge.totalDocs.toString(); pillColor: Theme.knowledge }
                        StatPill { label: "Chunks"; value: Knowledge.totalChunks.toString(); pillColor: Theme.primary }
                    }
                }
            }

            /* ─── Content ─── */
            Item {
                Layout.fillWidth: true; Layout.fillHeight: true

                /* ─── Documents Tab ─── */
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 12
                    visible: activeTab === 0

                    RowLayout {
                        Layout.fillWidth: true; spacing: 8

                        Rectangle {
                            Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                            color: reindexMa.containsMouse ? Qt.darker(Theme.knowledge, 1.2) : Theme.knowledge
                            Text { anchors.centerIn: parent; text: "Reindex All"; color: "white"; font.pixelSize: 12; font.bold: true }
                            MouseArea {
                                id: reindexMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: Knowledge.reindex()
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true; height: 34; radius: Theme.radiusSmall
                            color: addDocMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt
                            Text { anchors.centerIn: parent; text: "+ Add Document"; color: Theme.text; font.pixelSize: 12 }
                            MouseArea {
                                id: addDocMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: Knowledge.addDocument("/docs/new_document.md", "New Document")
                            }
                        }
                    }

                    ListView {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        model: Knowledge.documents; clip: true; spacing: 4

                        delegate: Rectangle {
                            width: parent ? parent.width : 0
                            height: 64; radius: Theme.radiusTiny
                            color: docMa.containsMouse ? Theme.glassHover : Theme.surface
                            border.width: 1; border.color: Theme.glassBorder

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 12; spacing: 12

                                /* Type icon */
                                Rectangle {
                                    width: 36; height: 36; radius: Theme.radiusTiny
                                    color: Qt.rgba(Theme.knowledge.r, Theme.knowledge.g, Theme.knowledge.b, 0.15)
                                    Text {
                                        anchors.centerIn: parent
                                        text: {
                                            var t = modelData.type
                                            if (t === "md") return "\u{1F4DD}"
                                            if (t === "code") return "\u{1F4BB}"
                                            if (t === "pdf") return "\u{1F4C4}"
                                            if (t === "config") return "\u2699"
                                            return "\u{1F4C1}"
                                        }
                                        font.pixelSize: 16
                                    }
                                }

                                ColumnLayout {
                                    spacing: 2; Layout.fillWidth: true
                                    Text {
                                        text: modelData.title; color: Theme.text
                                        font.pixelSize: 12; font.bold: true
                                        Layout.fillWidth: true; elide: Text.ElideRight
                                    }
                                    Text {
                                        text: modelData.path; color: Theme.textDim
                                        font.pixelSize: 10; Layout.fillWidth: true; elide: Text.ElideRight
                                    }
                                }

                                ColumnLayout {
                                    spacing: 2
                                    Text { text: modelData.chunks + " chunks"; color: Theme.textDim; font.pixelSize: 10 }
                                    Text { text: modelData.type.toUpperCase(); color: Theme.knowledge; font.pixelSize: 9; font.bold: true }
                                }
                            }

                            MouseArea {
                                id: docMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            }
                        }
                    }
                }

                /* ─── Search Tab ─── */
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 12
                    visible: activeTab === 1

                    Rectangle {
                        Layout.fillWidth: true; height: 42; radius: Theme.radiusSmall
                        color: Theme.surfaceAlt; border.width: 1
                        border.color: kbSearchInput.activeFocus ? Theme.knowledge : Theme.glassBorder

                        RowLayout {
                            anchors.fill: parent; anchors.margins: 12; spacing: 8

                            TextInput {
                                id: kbSearchInput; Layout.fillWidth: true
                                color: Theme.text; font.pixelSize: 13; font.family: Theme.fontFamily
                                clip: true; selectByMouse: true

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "Search knowledge base..."; color: Theme.textMuted
                                    font.pixelSize: 13; visible: !kbSearchInput.text
                                }

                                Keys.onReturnPressed: {
                                    searchResults = Knowledge.search(text)
                                }
                            }

                            Rectangle {
                                width: 60; height: 28; radius: Theme.radiusTiny
                                color: searchBtnMa.containsMouse ? Qt.darker(Theme.knowledge, 1.2) : Theme.knowledge
                                Text { anchors.centerIn: parent; text: "Search"; color: "white"; font.pixelSize: 11 }
                                MouseArea {
                                    id: searchBtnMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: searchResults = Knowledge.search(kbSearchInput.text)
                                }
                            }
                        }
                    }

                    ListView {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        model: searchResults; clip: true; spacing: 4

                        delegate: Rectangle {
                            width: parent ? parent.width : 0
                            height: 72; radius: Theme.radiusTiny
                            color: srMa.containsMouse ? Theme.glassHover : Theme.surface
                            border.width: 1; border.color: Theme.glassBorder

                            ColumnLayout {
                                anchors.fill: parent; anchors.margins: 12; spacing: 4

                                RowLayout {
                                    spacing: 8
                                    Text {
                                        text: modelData.title; color: Theme.knowledge
                                        font.pixelSize: 12; font.bold: true
                                        Layout.fillWidth: true
                                    }
                                    Text {
                                        text: (modelData.relevance * 100).toFixed(0) + "% match"
                                        color: Theme.success; font.pixelSize: 10; font.bold: true
                                    }
                                }
                                Text {
                                    text: modelData.snippet; color: Theme.textDim
                                    font.pixelSize: 11; Layout.fillWidth: true
                                    maximumLineCount: 2; elide: Text.ElideRight
                                    wrapMode: Text.WordWrap
                                }
                            }

                            MouseArea {
                                id: srMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            text: "Enter a search query and press Enter"
                            color: Theme.textMuted; font.pixelSize: 12
                            visible: searchResults.length === 0
                        }
                    }
                }

                /* ─── Ask AI Tab ─── */
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 12
                    visible: activeTab === 2

                    Text {
                        text: "Ask a question about your knowledge base"
                        color: Theme.text; font.pixelSize: 14; font.bold: true
                    }

                    Rectangle {
                        Layout.fillWidth: true; height: 42; radius: Theme.radiusSmall
                        color: Theme.surfaceAlt; border.width: 1
                        border.color: qaInput.activeFocus ? Theme.knowledge : Theme.glassBorder

                        RowLayout {
                            anchors.fill: parent; anchors.margins: 12; spacing: 8

                            TextInput {
                                id: qaInput; Layout.fillWidth: true
                                color: Theme.text; font.pixelSize: 13; font.family: Theme.fontFamily
                                clip: true; selectByMouse: true

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "Ask a question..."; color: Theme.textMuted
                                    font.pixelSize: 13; visible: !qaInput.text
                                }

                                Keys.onReturnPressed: {
                                    qaAnswer = Knowledge.askQuestion(text)
                                }
                            }

                            Rectangle {
                                width: 50; height: 28; radius: Theme.radiusTiny
                                color: askBtnMa.containsMouse ? Qt.darker(Theme.knowledge, 1.2) : Theme.knowledge
                                Text { anchors.centerIn: parent; text: "Ask"; color: "white"; font.pixelSize: 11 }
                                MouseArea {
                                    id: askBtnMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: qaAnswer = Knowledge.askQuestion(qaInput.text)
                                }
                            }
                        }
                    }

                    /* Answer display */
                    Rectangle {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        radius: Theme.radiusSmall; color: Theme.surface
                        border.width: 1; border.color: Theme.glassBorder

                        ScrollView {
                            anchors.fill: parent; anchors.margins: 16
                            clip: true

                            Text {
                                width: parent.width
                                text: qaAnswer || "Your AI-powered answer will appear here..."
                                color: qaAnswer ? Theme.text : Theme.textMuted
                                font.pixelSize: 13; wrapMode: Text.WordWrap
                                lineHeight: 1.5
                            }
                        }
                    }
                }
            }
        }
    }

    component StatPill: Rectangle {
        property string label: ""; property string value: ""; property color pillColor: Theme.primary
        width: spRow.width + 16; height: 24; radius: 12
        color: Qt.rgba(pillColor.r, pillColor.g, pillColor.b, 0.12)
        Row { id: spRow; anchors.centerIn: parent; spacing: 4
            Text { text: parent.parent.label; color: Theme.textDim; font.pixelSize: 10 }
            Text { text: parent.parent.value; color: parent.parent.pillColor; font.pixelSize: 10; font.bold: true }
        }
    }
}
