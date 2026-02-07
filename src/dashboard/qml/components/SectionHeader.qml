import QtQuick 2.15
import ".."

Item {
    width: parent ? parent.width : 200
    height: 32

    property string title: ""
    property color accentColor: Theme.primary

    Row {
        anchors.verticalCenter: parent.verticalCenter
        spacing: 8

        Rectangle {
            width: 3; height: 18
            radius: 2
            color: accentColor
            anchors.verticalCenter: parent.verticalCenter
        }

        Text {
            text: title
            color: Theme.text
            font.pixelSize: Theme.fontSizeLarge
            font.bold: true
        }
    }
}
