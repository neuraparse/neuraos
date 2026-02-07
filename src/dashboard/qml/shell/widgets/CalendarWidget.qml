import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 210; height: 150
    widgetTitle: "Calendar"

    ColumnLayout {
        anchors.fill: parent
        spacing: 6

        /* Today header */
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Rectangle {
                width: 40; height: 40
                radius: Theme.radiusTiny
                color: Theme.primary

                Column {
                    anchors.centerIn: parent
                    spacing: -2

                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: {
                            var d = new Date()
                            return Qt.formatDate(d, "d")
                        }
                        font.pixelSize: 18
                        font.bold: true
                        font.family: Theme.fontFamily
                        color: "#FFFFFF"
                    }

                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: {
                            var d = new Date()
                            return Qt.formatDate(d, "MMM")
                        }
                        font.pixelSize: 9
                        font.family: Theme.fontFamily
                        color: Qt.rgba(1, 1, 1, 0.8)
                    }
                }
            }

            Column {
                spacing: 2
                Text {
                    text: Qt.formatDate(new Date(), "dddd")
                    font.pixelSize: 13
                    font.bold: true
                    font.family: Theme.fontFamily
                    color: Theme.text
                }
                Text {
                    text: Qt.formatDate(new Date(), "MMMM yyyy")
                    font.pixelSize: 10
                    font.family: Theme.fontFamily
                    color: Theme.textDim
                }
            }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

        /* Upcoming events */
        Repeater {
            model: [
                { time: "09:00", label: "Team Standup", clr: "#5B9AFF" },
                { time: "11:30", label: "AI Model Review", clr: "#A78BFA" },
                { time: "14:00", label: "Deploy v4.0", clr: "#34D399" }
            ]

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Rectangle {
                    width: 3; height: 16; radius: 1.5
                    color: modelData.clr
                }

                Text {
                    text: modelData.time
                    font.pixelSize: 10
                    font.family: Theme.fontFamily
                    font.weight: Font.DemiBold
                    color: Theme.textDim
                    Layout.preferredWidth: 34
                }

                Text {
                    Layout.fillWidth: true
                    text: modelData.label
                    font.pixelSize: 11
                    font.family: Theme.fontFamily
                    color: Theme.text
                    elide: Text.ElideRight
                }
            }
        }
    }
}
