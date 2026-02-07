import QtQuick 2.15
import ".."

Rectangle {
    id: btn
    width: 48; height: 48
    radius: Theme.radiusSmall
    color: mouseArea.containsMouse ? Theme.surfaceAlt : "transparent"

    property string icon: ""
    property string label: ""
    property color iconColor: Theme.text
    signal clicked()

    Column {
        anchors.centerIn: parent
        spacing: 2

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: btn.icon
            font.pixelSize: 20
            color: btn.iconColor
        }

        Text {
            visible: btn.label !== ""
            anchors.horizontalCenter: parent.horizontalCenter
            text: btn.label
            font.pixelSize: 9
            color: Theme.textDim
        }
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: btn.clicked()
    }

    Behavior on color { ColorAnimation { duration: Theme.animFast } }
}
