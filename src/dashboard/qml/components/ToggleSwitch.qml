import QtQuick 2.15
import ".."

Item {
    id: toggle
    width: 48; height: 26

    property bool checked: false
    signal toggled(bool value)

    Rectangle {
        anchors.fill: parent
        radius: height / 2
        color: toggle.checked ? Theme.primary : Theme.surfaceLight

        Behavior on color { ColorAnimation { duration: Theme.animNormal } }

        Rectangle {
            id: knob
            width: 20; height: 20
            radius: 10
            anchors.verticalCenter: parent.verticalCenter
            x: toggle.checked ? parent.width - width - 3 : 3
            color: "#FFFFFF"

            Behavior on x { NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutQuad } }
        }
    }

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: {
            toggle.checked = !toggle.checked
            toggle.toggled(toggle.checked)
        }
    }
}
