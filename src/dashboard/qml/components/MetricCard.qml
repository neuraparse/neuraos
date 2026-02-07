import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."

Rectangle {
    id: card
    width: 180; height: 90
    color: Theme.surface
    radius: Theme.radius
    border.width: 1
    border.color: Theme.surfaceLight

    property string title: ""
    property string value: ""
    property string unit: ""
    property color accentColor: Theme.primary
    property string icon: ""

    RowLayout {
        anchors.fill: parent
        anchors.margins: Theme.cardPadding
        spacing: 12

        /* Icon circle */
        Rectangle {
            width: 40; height: 40
            radius: 20
            color: Qt.rgba(card.accentColor.r, card.accentColor.g, card.accentColor.b, 0.15)

            Text {
                anchors.centerIn: parent
                text: card.icon
                font.pixelSize: 18
                color: card.accentColor
            }
        }

        Column {
            Layout.fillWidth: true
            spacing: 2

            Text {
                text: card.title
                color: Theme.textDim
                font.pixelSize: Theme.fontSizeSmall
            }

            Row {
                id: valueRow
                spacing: 4
                Text {
                    text: card.value
                    color: Theme.text
                    font.pixelSize: Theme.fontSizeXL
                    font.bold: true
                }
                Text {
                    text: card.unit
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeSmall
                    y: valueRow.height - height - 4
                }
            }
        }
    }

    /* Hover effect */
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onEntered: card.border.color = card.accentColor
        onExited: card.border.color = Theme.surfaceLight
    }

    Behavior on border.color { ColorAnimation { duration: Theme.animFast } }
}
