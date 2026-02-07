import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: memoryApp
    anchors.fill: parent

    property string selectedCategory: ""
    property string searchQuery: ""

    function filteredEntries() {
        var all = AIMemory.entries
        var result = []
        for (var i = 0; i < all.length; i++) {
            var e = all[i]
            if (selectedCategory !== "" && e.category !== selectedCategory) continue
            if (searchQuery !== "" &&
                e.key.toLowerCase().indexOf(searchQuery.toLowerCase()) === -1 &&
                e.value.toLowerCase().indexOf(searchQuery.toLowerCase()) === -1) continue
            result.push(e)
        }
        return result
    }

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

                    Text { text: "AI Memory"; color: Theme.aiMemory; font.pixelSize: 15; font.bold: true }

                    Item { Layout.fillWidth: true }

                    Row {
                        spacing: 12
                        StatPill { label: "Entries"; value: AIMemory.memoryEntries.toString(); pillColor: Theme.aiMemory }
                        StatPill { label: "Size"; value: (AIMemory.totalSize / 1024).toFixed(1) + " KB"; pillColor: Theme.primary }
                        StatPill { label: "Categories"; value: AIMemory.categories.length.toString(); pillColor: Theme.success }
                    }
                }
            }

            /* ─── Content ─── */
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; spacing: 0

                /* ─── Categories Sidebar ─── */
                Rectangle {
                    Layout.preferredWidth: 200; Layout.fillHeight: true; color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 6

                        Text { text: "Categories"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                        /* All category */
                        Rectangle {
                            Layout.fillWidth: true; height: 34; radius: Theme.radiusTiny
                            color: selectedCategory === "" ? Theme.glassActive : allCatMa.containsMouse ? Theme.glassHover : "transparent"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter; x: 12
                                text: "All (" + AIMemory.memoryEntries + ")"
                                color: selectedCategory === "" ? Theme.aiMemory : Theme.text
                                font.pixelSize: 12
                            }
                            MouseArea {
                                id: allCatMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: selectedCategory = ""
                            }
                        }

                        Repeater {
                            model: AIMemory.categories

                            Rectangle {
                                Layout.fillWidth: true; height: 34; radius: Theme.radiusTiny
                                color: selectedCategory === modelData ? Theme.glassActive :
                                       catMa.containsMouse ? Theme.glassHover : "transparent"

                                Row {
                                    anchors.verticalCenter: parent.verticalCenter; x: 12; spacing: 8

                                    Rectangle {
                                        width: 8; height: 8; radius: 4
                                        color: {
                                            if (modelData === "conversations") return Theme.primary
                                            if (modelData === "preferences") return Theme.secondary
                                            if (modelData === "tasks") return Theme.warning
                                            if (modelData === "knowledge") return Theme.success
                                            return Theme.textDim
                                        }
                                    }
                                    Text {
                                        text: modelData
                                        color: selectedCategory === modelData ? Theme.aiMemory : Theme.text
                                        font.pixelSize: 12
                                    }
                                }
                                MouseArea {
                                    id: catMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedCategory = modelData
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }

                        /* Clear button */
                        Rectangle {
                            Layout.fillWidth: true; height: 32; radius: Theme.radiusSmall
                            color: clearAllMa.containsMouse ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.2) : "transparent"
                            border.width: 1; border.color: Theme.error

                            Text { anchors.centerIn: parent; text: "Clear All"; color: Theme.error; font.pixelSize: 11 }
                            MouseArea {
                                id: clearAllMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: AIMemory.clearAll()
                            }
                        }
                    }
                }

                Rectangle { Layout.fillHeight: true; width: 1; color: Theme.glassBorder }

                /* ─── Entries Panel ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true; color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        /* Search */
                        Rectangle {
                            Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                            color: Theme.surfaceAlt; border.width: 1
                            border.color: memSearchInput.activeFocus ? Theme.aiMemory : Theme.glassBorder

                            TextInput {
                                id: memSearchInput
                                anchors.fill: parent; anchors.margins: 10
                                color: Theme.text; font.pixelSize: 12; font.family: Theme.fontFamily
                                clip: true; selectByMouse: true
                                onTextChanged: searchQuery = text

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "Search memory entries..."
                                    color: Theme.textMuted; font.pixelSize: 12
                                    visible: !memSearchInput.text && !memSearchInput.activeFocus
                                }
                            }
                        }

                        /* Entry list */
                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: filteredEntries()
                            clip: true; spacing: 4

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: 72; radius: Theme.radiusTiny
                                color: entryMa.containsMouse ? Theme.glassHover : Theme.surface
                                border.width: 1; border.color: Theme.glassBorder

                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 3

                                    RowLayout {
                                        spacing: 8
                                        Rectangle {
                                            width: 8; height: 8; radius: 4
                                            color: {
                                                if (modelData.category === "conversations") return Theme.primary
                                                if (modelData.category === "preferences") return Theme.secondary
                                                if (modelData.category === "tasks") return Theme.warning
                                                if (modelData.category === "knowledge") return Theme.success
                                                return Theme.textDim
                                            }
                                        }
                                        Text {
                                            text: modelData.key; color: Theme.text
                                            font.pixelSize: 12; font.bold: true
                                            Layout.fillWidth: true; elide: Text.ElideRight
                                        }
                                        Text {
                                            text: modelData.source; color: Theme.textDim
                                            font.pixelSize: 10
                                        }
                                    }

                                    Text {
                                        text: modelData.value; color: Theme.textDim
                                        font.pixelSize: 11; Layout.fillWidth: true
                                        elide: Text.ElideRight; maximumLineCount: 1
                                    }

                                    RowLayout {
                                        spacing: 8
                                        Text { text: modelData.timestamp; color: Theme.textMuted; font.pixelSize: 9 }
                                        Text { text: modelData.category; color: Theme.textMuted; font.pixelSize: 9 }
                                        Item { Layout.fillWidth: true }
                                        Text { text: "x" + modelData.accessCount + " accessed"; color: Theme.textMuted; font.pixelSize: 9 }
                                    }
                                }

                                MouseArea {
                                    id: entryMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                }
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
