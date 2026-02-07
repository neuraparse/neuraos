import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."

Rectangle {
    id: notifItem
    width: parent ? parent.width : 300
    height: 64
    radius: Theme.radiusSmall
    color: notifMa.containsMouse ? Theme.surfaceAlt : Theme.surface

    property string title: ""
    property string message: ""
    property string timestamp: ""
    property string icon: "\u2709"
    property color iconColor: Theme.primary

    signal dismissed()

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        Rectangle {
            width: 32; height: 32; radius: 16
            color: Qt.rgba(iconColor.r, iconColor.g, iconColor.b, 0.15)
            CanvasIcon { anchors.centerIn: parent; iconName: notifItem.icon; iconSize: 14; iconColor: notifItem.iconColor }
        }

        Column {
            Layout.fillWidth: true
            spacing: 2

            Text { text: title; color: Theme.text; font.pixelSize: 11; font.bold: true; elide: Text.ElideRight; width: parent.width }
            Text { text: message; color: Theme.textDim; font.pixelSize: 10; elide: Text.ElideRight; width: parent.width }
        }

        Column {
            spacing: 4

            Text { text: timestamp; color: Theme.textMuted; font.pixelSize: 9 }

            Rectangle {
                width: 18; height: 18; radius: 9
                color: dismissMa.containsMouse ? Theme.error : "transparent"
                anchors.horizontalCenter: parent.horizontalCenter

                Text { anchors.centerIn: parent; text: "\u2715"; font.pixelSize: 10; color: Theme.textDim }
                MouseArea {
                    id: dismissMa; anchors.fill: parent
                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: notifItem.dismissed()
                }
            }
        }
    }

    MouseArea {
        id: notifMa
        anchors.fill: parent
        hoverEnabled: true
        z: -1
    }
}
