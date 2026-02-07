import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: launcherPage

    property string searchText: ""
    property int selectedCategory: -1  /* -1 = All */

    ListModel {
        id: appsModel

        /* AI Tools */
        ListElement { name: "AI Dashboard";   icon: "\u2699"; category: "AI Tools";  clr: "#00D9FF"; page: 1 }
        ListElement { name: "Model Manager";  icon: "\u2261"; category: "AI Tools";  clr: "#7C3AED"; page: 1 }
        ListElement { name: "NPIE Bench";     icon: "\u23F1"; category: "AI Tools";  clr: "#10B981"; page: -1 }
        ListElement { name: "NPU Monitor";    icon: "\u2756"; category: "AI Tools";  clr: "#F59E0B"; page: 1 }
        ListElement { name: "Tensor Viewer";  icon: "\u25A6"; category: "AI Tools";  clr: "#EC4899"; page: -1 }
        ListElement { name: "Data Pipeline";  icon: "\u21C4"; category: "AI Tools";  clr: "#06B6D4"; page: -1 }

        /* System */
        ListElement { name: "Terminal";       icon: "\u2756"; category: "System";    clr: "#10B981"; page: 3 }
        ListElement { name: "File Manager";   icon: "\u2302"; category: "System";    clr: "#3B82F6"; page: 4 }
        ListElement { name: "System Monitor"; icon: "\u2261"; category: "System";    clr: "#F59E0B"; page: 2 }
        ListElement { name: "Process List";   icon: "\u2630"; category: "System";    clr: "#EF4444"; page: 2 }
        ListElement { name: "Logs Viewer";    icon: "\u2263"; category: "System";    clr: "#8B5CF6"; page: -1 }
        ListElement { name: "Disk Usage";     icon: "\u25C9"; category: "System";    clr: "#F97316"; page: 2 }

        /* Settings */
        ListElement { name: "Settings";       icon: "\u2731"; category: "Settings";  clr: "#7C3AED"; page: 5 }
        ListElement { name: "Network";        icon: "\u2301"; category: "Settings";  clr: "#06B6D4"; page: 5 }
        ListElement { name: "Display";        icon: "\u25A3"; category: "Settings";  clr: "#F59E0B"; page: 5 }
        ListElement { name: "Power";          icon: "\u26A1"; category: "Settings";  clr: "#EF4444"; page: -1 }

        /* Utilities */
        ListElement { name: "Calculator";     icon: "\u2328"; category: "Utilities"; clr: "#64748B"; page: -1 }
        ListElement { name: "Text Editor";    icon: "\u270E"; category: "Utilities"; clr: "#94A3B8"; page: -1 }
        ListElement { name: "Clock";          icon: "\u231A"; category: "Utilities"; clr: "#A78BFA"; page: 0 }
        ListElement { name: "Screenshot";     icon: "\u2702"; category: "Utilities"; clr: "#FB923C"; page: -1 }
    }

    property var categories: ["AI Tools", "System", "Settings", "Utilities"]

    function matchesFilter(index) {
        var item = appsModel.get(index)
        if (searchText !== "" && item.name.toLowerCase().indexOf(searchText.toLowerCase()) === -1)
            return false
        if (selectedCategory >= 0 && item.category !== categories[selectedCategory])
            return false
        return true
    }

    Flickable {
        anchors.fill: parent
        anchors.margins: 20
        contentHeight: col.height
        clip: true

        ColumnLayout {
            id: col
            width: parent.width
            spacing: 16

            Components.SectionHeader { title: "Applications" }

            /* Search bar */
            Components.SearchBar {
                Layout.fillWidth: true
                placeholder: "Search apps..."
                onTextChanged: searchText = text
            }

            /* Category filter */
            Rectangle {
                Layout.fillWidth: true
                height: 42
                radius: Theme.radius
                color: Theme.surface

                Row {
                    anchors.centerIn: parent
                    spacing: 6

                    /* All button */
                    Rectangle {
                        width: 60; height: 30
                        radius: Theme.radiusSmall
                        color: selectedCategory === -1 ? Theme.primary : (allMa.containsMouse ? Theme.surfaceAlt : "transparent")

                        Text {
                            anchors.centerIn: parent
                            text: "All"
                            color: selectedCategory === -1 ? "#000000" : Theme.text
                            font.pixelSize: Theme.fontSizeSmall
                            font.bold: selectedCategory === -1
                        }

                        MouseArea {
                            id: allMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: selectedCategory = -1
                        }
                    }

                    Repeater {
                        model: categories

                        Rectangle {
                            width: 90; height: 30
                            radius: Theme.radiusSmall
                            color: selectedCategory === index ? Theme.primary : (catBtnMa.containsMouse ? Theme.surfaceAlt : "transparent")

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                color: selectedCategory === index ? "#000000" : Theme.text
                                font.pixelSize: Theme.fontSizeSmall
                                font.bold: selectedCategory === index
                            }

                            MouseArea {
                                id: catBtnMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: selectedCategory = index
                            }
                        }
                    }
                }
            }

            /* App Grid */
            Rectangle {
                Layout.fillWidth: true
                height: appGrid.height + 24
                radius: Theme.radius
                color: Theme.surface

                Flow {
                    id: appGrid
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.margins: 12
                    spacing: 12

                    Repeater {
                        model: appsModel.count

                        Rectangle {
                            width: 110; height: 110
                            radius: Theme.radius
                            color: appMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            border.width: 1
                            border.color: appMa.containsMouse ? appsModel.get(index).clr : "transparent"
                            visible: matchesFilter(index)

                            Column {
                                anchors.centerIn: parent
                                spacing: 8

                                /* Icon circle */
                                Rectangle {
                                    width: 48; height: 48; radius: 24
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    color: Qt.rgba(0, 0, 0, 0)
                                    border.width: 2
                                    border.color: appsModel.get(index).clr

                                    Text {
                                        anchors.centerIn: parent
                                        text: appsModel.get(index).icon
                                        font.pixelSize: 22
                                        color: appsModel.get(index).clr
                                    }
                                }

                                /* App name */
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    width: 100
                                    text: appsModel.get(index).name
                                    color: Theme.text
                                    font.pixelSize: 11
                                    horizontalAlignment: Text.AlignHCenter
                                    elide: Text.ElideRight
                                }

                                /* Category */
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: appsModel.get(index).category
                                    color: Theme.textDim
                                    font.pixelSize: 9
                                }
                            }

                            MouseArea {
                                id: appMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    var pg = appsModel.get(index).page
                                    if (pg >= 0) {
                                        launcherPage.parent.parent.currentIndex = pg
                                    }
                                }
                            }

                            Behavior on border.color { ColorAnimation { duration: Theme.animFast } }
                        }
                    }
                }
            }

            /* App count */
            Text {
                Layout.alignment: Qt.AlignRight
                text: {
                    var count = 0
                    for (var i = 0; i < appsModel.count; i++) {
                        if (matchesFilter(i)) count++
                    }
                    return count + " applications"
                }
                color: Theme.textDim
                font.pixelSize: Theme.fontSizeSmall
            }

            Item { height: 20 }
        }
    }
}
