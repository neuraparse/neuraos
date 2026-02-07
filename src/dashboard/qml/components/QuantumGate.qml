import QtQuick 2.15
import ".."

Rectangle {
    id: qGate
    width: 48; height: 48
    radius: 6
    color: isActive ? Qt.rgba(gateColor.r, gateColor.g, gateColor.b, 0.2) : Theme.surfaceAlt
    border.width: 1
    border.color: isActive ? gateColor : Theme.surfaceLight

    property string gateName: "H"
    property color gateColor: Theme.primary
    property bool isActive: false

    signal clicked()

    Text {
        anchors.centerIn: parent
        text: gateName
        color: isActive ? gateColor : Theme.text
        font.pixelSize: 16
        font.bold: true
        font.family: "monospace"
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: qGate.clicked()
    }

    Behavior on color { ColorAnimation { duration: Theme.animFast } }
    Behavior on border.color { ColorAnimation { duration: Theme.animFast } }
}
