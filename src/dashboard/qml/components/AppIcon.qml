import QtQuick 2.15
import ".."

Item {
    id: appIcon
    width: 72; height: 80

    property string iconText: "\u25A3"
    property string title: "App"
    property color iconColor: Theme.primary
    property bool showTitle: true

    signal clicked()

    Rectangle {
        id: iconBg
        anchors.horizontalCenter: parent.horizontalCenter
        width: 48; height: 48
        radius: 14
        color: iconMa.containsMouse ?
            Qt.rgba(iconColor.r, iconColor.g, iconColor.b, 0.25) :
            Qt.rgba(iconColor.r, iconColor.g, iconColor.b, 0.12)

        Text {
            anchors.centerIn: parent
            text: iconText
            font.pixelSize: 22
            color: iconColor
        }

        Behavior on color { ColorAnimation { duration: Theme.animFast } }
    }

    Text {
        anchors.top: iconBg.bottom
        anchors.topMargin: 4
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width
        text: title
        color: Theme.text
        font.pixelSize: 10
        horizontalAlignment: Text.AlignHCenter
        elide: Text.ElideRight
        visible: showTitle
    }

    MouseArea {
        id: iconMa
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: appIcon.clicked()
    }
}
