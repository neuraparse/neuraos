import QtQuick 2.15
import ".."

Rectangle {
    id: badge
    width: label.implicitWidth + 16
    height: 24
    radius: 12
    color: Qt.rgba(badgeColor.r, badgeColor.g, badgeColor.b, 0.15)

    property string text: ""
    property color badgeColor: Theme.success

    Text {
        id: label
        anchors.centerIn: parent
        text: badge.text
        color: badge.badgeColor
        font.pixelSize: Theme.fontSizeSmall
        font.bold: true
    }
}
