import QtQuick 2.15
import ".."

Rectangle {
    id: searchBar
    height: 42
    radius: Theme.radiusSmall
    color: Theme.surfaceAlt
    border.width: input.activeFocus ? 1.5 : 0
    border.color: Theme.primary

    property alias text: input.text
    property string placeholder: "Search..."

    /* Focus glow effect */
    Rectangle {
        visible: input.activeFocus
        anchors.fill: parent; anchors.margins: -2
        radius: parent.radius + 2
        color: "transparent"
        border.width: 2
        border.color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
        Behavior on opacity { NumberAnimation { duration: Theme.animFast } }
    }

    Row {
        anchors.fill: parent
        anchors.leftMargin: 14
        anchors.rightMargin: 14
        spacing: 10

        CanvasIcon {
            anchors.verticalCenter: parent.verticalCenter
            iconName: "search"
            iconColor: input.activeFocus ? Theme.primary : Theme.textDim
            iconSize: 16
            Behavior on iconColor { ColorAnimation { duration: Theme.animFast } }
        }

        TextInput {
            id: input
            anchors.verticalCenter: parent.verticalCenter
            width: parent.width - 40
            color: Theme.text
            font.pixelSize: Theme.fontSizeNormal
            font.family: Theme.fontFamily
            clip: true
            selectByMouse: true
            selectionColor: Theme.primary

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: searchBar.placeholder
                color: Theme.textMuted
                font.pixelSize: Theme.fontSizeNormal
                font.family: Theme.fontFamily
                visible: !input.text && !input.activeFocus
            }
        }
    }

    Behavior on border.width { NumberAnimation { duration: Theme.animFast } }
}
