import QtQuick 2.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 210; height: 110
    widgetTitle: ""

    Column {
        anchors.centerIn: parent
        spacing: 4

        Text {
            id: timeText
            anchors.horizontalCenter: parent.horizontalCenter
            font.pixelSize: 38
            font.weight: Font.Bold
            font.family: Theme.fontFamily
            color: Theme.text

            Timer {
                interval: 1000; running: true; repeat: true; triggeredOnStart: true
                onTriggered: {
                    var d = new Date()
                    timeText.text = Qt.formatTime(d, "HH:mm")
                    secText.text = Qt.formatTime(d, "ss")
                    dateText.text = Qt.formatDate(d, "dddd, MMMM d")
                }
            }
        }

        /* Seconds + date row */
        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 6

            Text {
                id: secText
                font.pixelSize: 14
                font.family: Theme.fontFamily
                color: Theme.primary
                font.weight: Font.DemiBold
            }

            Rectangle { width: 1; height: 14; color: Theme.surfaceLight; anchors.verticalCenter: parent.verticalCenter }

            Text {
                id: dateText
                font.pixelSize: 12
                font.family: Theme.fontFamily
                color: Theme.textDim
            }
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: SystemInfo.uptime
            font.pixelSize: 9
            font.family: Theme.fontFamily
            color: Theme.textMuted
        }
    }
}
