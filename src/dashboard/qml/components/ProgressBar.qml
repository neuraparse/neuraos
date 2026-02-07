import QtQuick 2.15
import ".."

Item {
    id: bar
    height: 4
    property real value: 0
    property real maxValue: 100
    property color barColor: Theme.primary
    property bool indeterminate: false

    Rectangle {
        anchors.fill: parent
        radius: 2
        color: Theme.surfaceLight
    }

    Rectangle {
        height: parent.height
        width: indeterminate ? parent.width * 0.3 : parent.width * Math.min(1, value / maxValue)
        radius: 2
        color: barColor

        Behavior on width { NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutQuint } }

        SequentialAnimation on x {
            running: indeterminate; loops: Animation.Infinite
            NumberAnimation { from: -bar.width * 0.3; to: bar.width; duration: 1200; easing.type: Easing.InOutQuad }
        }
    }
}
