import QtQuick 2.15
import ".."

Canvas {
    id: miniChart
    width: 120; height: 40

    property var data: []
    property color lineColor: Theme.primary
    property real maxValue: 100

    onDataChanged: requestPaint()

    onPaint: {
        var ctx = getContext("2d")
        ctx.clearRect(0, 0, width, height)

        if (data.length < 2) return

        var step = width / (data.length - 1)
        var mv = maxValue > 0 ? maxValue : 1

        ctx.beginPath()
        ctx.moveTo(0, height - (data[0] / mv * height))
        for (var i = 1; i < data.length; i++) {
            ctx.lineTo(i * step, height - (data[i] / mv * height))
        }
        ctx.strokeStyle = lineColor
        ctx.lineWidth = 1.5
        ctx.stroke()

        /* Fill under */
        ctx.lineTo((data.length - 1) * step, height)
        ctx.lineTo(0, height)
        ctx.closePath()
        ctx.fillStyle = Qt.rgba(lineColor.r, lineColor.g, lineColor.b, 0.08)
        ctx.fill()
    }
}
