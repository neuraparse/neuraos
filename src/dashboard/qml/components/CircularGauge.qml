import QtQuick 2.15
import ".."

Item {
    id: gauge
    width: 100; height: 100

    property real value: 0       /* 0-100 */
    property color gaugeColor: Theme.primary
    property string label: ""
    property string unit: "%"

    Canvas {
        id: canvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            var cx = width / 2
            var cy = height / 2
            var r = Math.min(cx, cy) - 6
            var startAngle = Math.PI * 0.75
            var endAngle = Math.PI * 2.25
            var valueAngle = startAngle + (endAngle - startAngle) * (gauge.value / 100)

            ctx.clearRect(0, 0, width, height)

            /* Background arc */
            ctx.beginPath()
            ctx.arc(cx, cy, r, startAngle, endAngle)
            ctx.lineWidth = 6
            ctx.strokeStyle = Theme.surfaceLight
            ctx.lineCap = "round"
            ctx.stroke()

            /* Value arc */
            if (gauge.value > 0) {
                ctx.beginPath()
                ctx.arc(cx, cy, r, startAngle, valueAngle)
                ctx.lineWidth = 6
                ctx.strokeStyle = gauge.gaugeColor
                ctx.lineCap = "round"
                ctx.stroke()
            }
        }
    }

    /* Value text */
    Column {
        anchors.centerIn: parent
        spacing: 0

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: Math.round(gauge.value)
            color: gauge.gaugeColor
            font.pixelSize: gauge.width * 0.28
            font.bold: true
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: gauge.unit
            color: Theme.textDim
            font.pixelSize: gauge.width * 0.12
        }
    }

    /* Label below */
    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        text: gauge.label
        color: Theme.textDim
        font.pixelSize: Theme.fontSizeSmall
    }

    onValueChanged: canvas.requestPaint()
    Component.onCompleted: canvas.requestPaint()
}
