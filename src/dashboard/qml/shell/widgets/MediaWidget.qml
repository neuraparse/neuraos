import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 240; height: 100
    widgetTitle: "Now Playing"

    property bool isPlaying: false

    RowLayout {
        anchors.fill: parent
        spacing: 10

        /* Album art (gradient placeholder) */
        Rectangle {
            width: 56; height: 56
            radius: Theme.radiusTiny
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#5B9AFF" }
                GradientStop { position: 1.0; color: "#A78BFA" }
            }

            Components.CanvasIcon {
                anchors.centerIn: parent
                iconName: "music"
                iconSize: 20
                iconColor: "#FFFFFF"
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            Text {
                text: "Neural Dreams"
                font.pixelSize: 12
                font.bold: true
                font.family: Theme.fontFamily
                color: Theme.text
                elide: Text.ElideRight
                Layout.fillWidth: true
            }

            Text {
                text: "AI Orchestra"
                font.pixelSize: 10
                font.family: Theme.fontFamily
                color: Theme.textDim
            }

            /* Progress bar */
            Rectangle {
                Layout.fillWidth: true
                height: 3; radius: 1.5
                color: Theme.surfaceLight

                Rectangle {
                    width: parent.width * 0.42
                    height: parent.height; radius: 1.5
                    color: Theme.primary
                }
            }

            /* Controls */
            Row {
                spacing: 12

                Repeater {
                    model: [
                        { icon: "skip-back",  act: "prev" },
                        { icon: isPlaying ? "pause" : "play", act: "toggle" },
                        { icon: "skip-forward", act: "next" }
                    ]

                    Components.CanvasIcon {
                        iconName: modelData.icon
                        iconSize: modelData.act === "toggle" ? 16 : 12
                        iconColor: mCtrlMa.containsMouse ? Theme.primary : Theme.textDim

                        MouseArea {
                            id: mCtrlMa
                            anchors.fill: parent; anchors.margins: -4
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                if (modelData.act === "toggle") isPlaying = !isPlaying
                            }
                        }
                    }
                }
            }
        }
    }
}
