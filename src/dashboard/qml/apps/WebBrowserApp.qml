import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: browserApp
    anchors.fill: parent

    property int activeTab: 0
    property var browserTabs: [
        { title: "NeuralOS Home", url: "https://neuralos.dev" },
        { title: "Documentation", url: "https://docs.neuralos.dev" }
    ]
    property bool showBookmarks: false

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Tab Bar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 34
                color: Theme.surfaceAlt

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 4
                    anchors.rightMargin: 4
                    spacing: 2

                    Repeater {
                        model: browserTabs.length

                        Rectangle {
                            Layout.preferredWidth: Math.min(180, 160)
                            Layout.fillHeight: true
                            Layout.topMargin: 4
                            radius: Theme.radiusSmall
                            color: index === activeTab ? Theme.surface : "transparent"

                            Rectangle {
                                visible: index === activeTab
                                anchors.bottom: parent.bottom
                                anchors.left: parent.left
                                anchors.right: parent.right
                                height: 2
                                color: Theme.surface
                            }

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 6
                                spacing: 6

                                Components.CanvasIcon {
                                    iconName: "globe"
                                    iconSize: 12
                                    iconColor: Theme.textDim
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: browserTabs[index].title
                                    font.pixelSize: 11
                                    color: index === activeTab ? Theme.text : Theme.textDim
                                    elide: Text.ElideRight
                                }

                                Rectangle {
                                    width: 16; height: 16; radius: 8
                                    color: btCloseMa.containsMouse ? Theme.surfaceLight : "transparent"
                                    Text {
                                        anchors.centerIn: parent
                                        text: "×"; font.pixelSize: 11; color: Theme.textDim
                                    }
                                    MouseArea {
                                        id: btCloseMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }

                            MouseArea {
                                anchors.fill: parent; z: -1
                                cursorShape: Qt.PointingHandCursor
                                onClicked: activeTab = index
                            }
                        }
                    }

                    /* New Tab Button */
                    Rectangle {
                        Layout.preferredWidth: 28; Layout.preferredHeight: 28
                        Layout.alignment: Qt.AlignVCenter
                        radius: Theme.radiusSmall
                        color: newTabMa.containsMouse ? Theme.surfaceLight : "transparent"
                        Text {
                            anchors.centerIn: parent
                            text: "+"; font.pixelSize: 16; color: Theme.textDim
                        }
                        MouseArea {
                            id: newTabMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        }
                    }

                    Item { Layout.fillWidth: true }
                }
            }

            /* Navigation Bar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 38
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 4

                    /* Nav Buttons */
                    Repeater {
                        model: [
                            { icon: "arrow-left", tip: "Back" },
                            { icon: "arrow-right", tip: "Forward" },
                            { icon: "refresh", tip: "Refresh" }
                        ]

                        Rectangle {
                            Layout.preferredWidth: 30; Layout.preferredHeight: 28
                            radius: Theme.radiusSmall
                            color: navBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: modelData.icon; iconSize: 14; iconColor: Theme.textDim
                            }
                            MouseArea {
                                id: navBtnMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            }
                        }
                    }

                    /* URL Bar */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 28
                        radius: 14
                        color: Theme.surfaceAlt
                        border.width: urlInput.activeFocus ? 1 : 0
                        border.color: Theme.primary
                        Behavior on border.width { NumberAnimation { duration: 100 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            spacing: 6

                            Components.CanvasIcon {
                                iconName: "shield"
                                iconSize: 12
                                iconColor: Theme.success
                            }

                            TextInput {
                                id: urlInput
                                Layout.fillWidth: true
                                text: browserTabs[activeTab].url
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                                color: Theme.text
                                clip: true
                                selectByMouse: true
                                selectionColor: Theme.primary
                                selectedTextColor: "#FFFFFF"
                                verticalAlignment: TextInput.AlignVCenter
                                onAccepted: {
                                    var tabs = browserTabs.slice()
                                    tabs[activeTab] = { title: text.replace(/^https?:\/\//, "").split("/")[0], url: text }
                                    browserTabs = tabs
                                }
                            }
                        }
                    }

                    /* Bookmark Toggle */
                    Rectangle {
                        Layout.preferredWidth: 30; Layout.preferredHeight: 28
                        radius: Theme.radiusSmall
                        color: showBookmarks ? Theme.primary : bkmkMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "bookmark"
                            iconSize: 14
                            iconColor: showBookmarks ? "#FFFFFF" : Theme.textDim
                        }
                        MouseArea {
                            id: bkmkMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: showBookmarks = !showBookmarks
                        }
                    }

                    Rectangle {
                        Layout.preferredWidth: 30; Layout.preferredHeight: 28
                        radius: Theme.radiusSmall
                        color: menuBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "menu"
                            iconSize: 14
                            iconColor: Theme.textDim
                        }
                        MouseArea {
                            id: menuBtnMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Content Area */
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* Bookmarks Sidebar */
                Rectangle {
                    visible: showBookmarks
                    Layout.preferredWidth: 200
                    Layout.fillHeight: true
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4

                        Text {
                            text: "Bookmarks"
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            color: Theme.text
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                        Repeater {
                            model: [
                                { name: "NeuralOS Home", icon: "globe" },
                                { name: "Documentation", icon: "file" },
                                { name: "AI Models Hub", icon: "neural" },
                                { name: "Package Registry", icon: "box" },
                                { name: "Community Forum", icon: "users" },
                                { name: "GitHub", icon: "code" }
                            ]

                            Rectangle {
                                Layout.fillWidth: true
                                height: 30
                                radius: Theme.radiusSmall
                                color: bkmkItemMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 8
                                    spacing: 8

                                    Components.CanvasIcon {
                                        iconName: modelData.icon
                                        iconSize: 12
                                        iconColor: Theme.textDim
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.name
                                        font.pixelSize: 12
                                        color: Theme.text
                                        elide: Text.ElideRight
                                    }
                                }

                                MouseArea {
                                    id: bkmkItemMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }

                Rectangle {
                    visible: showBookmarks
                    Layout.preferredWidth: 1
                    Layout.fillHeight: true
                    color: Theme.surfaceLight
                }

                /* Web Page Content (Mock) */
                Flickable {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentHeight: webContent.height
                    clip: true
                    flickableDirection: Flickable.VerticalFlick

                    ColumnLayout {
                        id: webContent
                        width: parent.width
                        spacing: 0

                        /* Hero Section */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 280
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: "#0A1628" }
                                GradientStop { position: 1.0; color: "#162444" }
                            }

                            ColumnLayout {
                                anchors.centerIn: parent
                                spacing: 16

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "NeuralOS"
                                    font.pixelSize: 42
                                    font.weight: Font.Bold
                                    color: "#FFFFFF"
                                }

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "The AI-Native Operating System"
                                    font.pixelSize: 18
                                    color: "#94A3B8"
                                }

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "Built for the future of computing. Neural processing,\nquantum-ready architecture, and intelligent workflows."
                                    font.pixelSize: 13
                                    color: "#64748B"
                                    horizontalAlignment: Text.AlignHCenter
                                    lineHeight: 1.4
                                }

                                RowLayout {
                                    Layout.alignment: Qt.AlignHCenter
                                    spacing: 12
                                    Layout.topMargin: 8

                                    Rectangle {
                                        width: 130; height: 36
                                        radius: 18
                                        color: Theme.primary
                                        Text {
                                            anchors.centerIn: parent
                                            text: "Get Started"
                                            font.pixelSize: 13
                                            font.weight: Font.DemiBold
                                            color: "#FFFFFF"
                                        }
                                    }

                                    Rectangle {
                                        width: 130; height: 36
                                        radius: 18
                                        color: "transparent"
                                        border.width: 1; border.color: "#475569"
                                        Text {
                                            anchors.centerIn: parent
                                            text: "Learn More"
                                            font.pixelSize: 13
                                            color: "#94A3B8"
                                        }
                                    }
                                }
                            }
                        }

                        /* Features Section */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: featuresCol.height + 60
                            color: Theme.background

                            ColumnLayout {
                                id: featuresCol
                                anchors.top: parent.top
                                anchors.topMargin: 30
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width - 40, 700)
                                spacing: 16

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "Core Features"
                                    font.pixelSize: 22
                                    font.weight: Font.DemiBold
                                    color: Theme.text
                                }

                                GridLayout {
                                    Layout.fillWidth: true
                                    columns: 2
                                    rowSpacing: 12
                                    columnSpacing: 12

                                    Repeater {
                                        model: [
                                            { title: "Neural Processing Unit", desc: "Hardware-accelerated AI inference with dedicated NPU support", icon: "chip", clr: "#F59E0B" },
                                            { title: "Quantum Computing Lab", desc: "Simulate and visualize quantum circuits and algorithms", icon: "atom", clr: "#A78BFA" },
                                            { title: "AI Agent Framework", desc: "Deploy autonomous AI agents for complex workflows", icon: "robot", clr: "#EC4899" },
                                            { title: "Defense Grade Security", desc: "Military-grade encryption and threat monitoring", icon: "shield", clr: "#EF4444" }
                                        ]

                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 100
                                            radius: Theme.radius
                                            color: Theme.surface
                                            border.width: 1
                                            border.color: Theme.surfaceLight

                                            ColumnLayout {
                                                anchors.fill: parent
                                                anchors.margins: 14
                                                spacing: 6

                                                RowLayout {
                                                    spacing: 8
                                                    Rectangle {
                                                        width: 28; height: 28; radius: 6
                                                        color: Qt.rgba(0, 0, 0, 0)
                                                        border.width: 1; border.color: modelData.clr

                                                        Components.CanvasIcon {
                                                            anchors.centerIn: parent
                                                            iconName: modelData.icon
                                                            iconSize: 14
                                                            iconColor: modelData.clr
                                                        }
                                                    }
                                                    Text {
                                                        text: modelData.title
                                                        font.pixelSize: 14
                                                        font.weight: Font.DemiBold
                                                        color: Theme.text
                                                    }
                                                }

                                                Text {
                                                    Layout.fillWidth: true
                                                    text: modelData.desc
                                                    font.pixelSize: 12
                                                    color: Theme.textDim
                                                    wrapMode: Text.WordWrap
                                                    lineHeight: 1.3
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /* Footer */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 60
                            color: Theme.surface

                            RowLayout {
                                anchors.centerIn: parent
                                spacing: 24

                                Text { text: "© 2025 NeuralOS Project"; font.pixelSize: 11; color: Theme.textDim }
                                Text { text: "Privacy"; font.pixelSize: 11; color: Theme.primary }
                                Text { text: "Terms"; font.pixelSize: 11; color: Theme.primary }
                                Text { text: "GitHub"; font.pixelSize: 11; color: Theme.primary }
                            }
                        }
                    }
                }
            }
        }
    }
}
