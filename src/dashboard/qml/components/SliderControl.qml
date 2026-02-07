import QtQuick 2.15
import ".."

Item {
    id: sliderControl
    height: 24
    implicitWidth: 200

    property real value: 50
    property real minValue: 0
    property real maxValue: 100
    property color trackColor: Theme.primary

    signal valueChanged(real newValue)

    Rectangle {
        anchors.verticalCenter: parent.verticalCenter
        width: parent.width
        height: 4
        radius: 2
        color: Theme.surfaceLight

        Rectangle {
            width: parent.width * (value - minValue) / (maxValue - minValue)
            height: parent.height
            radius: 2
            color: trackColor
        }
    }

    MouseArea {
        anchors.fill: parent
        onPressed: updateValue(mouse)
        onPositionChanged: if (pressed) updateValue(mouse)
        function updateValue(m) {
            var v = Math.max(minValue, Math.min(maxValue, minValue + (m.x / width) * (maxValue - minValue)))
            value = v
            sliderControl.valueChanged(v)
        }
    }

    Rectangle {
        x: (value - minValue) / (maxValue - minValue) * parent.width - 7
        anchors.verticalCenter: parent.verticalCenter
        width: 14; height: 14; radius: 7
        color: trackColor
    }
}
