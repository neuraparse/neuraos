import QtQuick 2.15
import ".."

Rectangle {
    id: card
    color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.7)
    radius: Theme.radius
    border.width: 1
    border.color: Theme.surfaceLight

    property alias contentItem: content.data

    default property alias data: content.data

    Item {
        id: content
        anchors.fill: parent
        anchors.margins: Theme.cardPadding
    }

    Behavior on scale { NumberAnimation { duration: Theme.animFast } }
}
