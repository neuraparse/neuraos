import QtQuick 2.15
import ".."

Rectangle {
    id: glass
    color: Theme.glass
    border.width: 1
    border.color: Theme.glassBorder
    radius: Theme.radiusSmall

    /* Top highlight edge for depth */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: parent.radius
        anchors.rightMargin: parent.radius
        height: 1
        color: Qt.rgba(1, 1, 1, Theme.darkMode ? 0.06 : 0.3)
    }
}
