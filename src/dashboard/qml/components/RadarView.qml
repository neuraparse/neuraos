import QtQuick 2.15
import ".."

Canvas {
    id: radarCanvas
    width: 200; height: 200

    property real sweepAngle: 0
    property var targets: []
    property color radarColor: Theme.success

    Timer {
        interval: 50; running: true; repeat: true
        onTriggered: {
            sweepAngle = (sweepAngle + 3) % 360
            radarCanvas.requestPaint()
        }
    }

    onPaint: {
        var ctx = getContext("2d")
        ctx.clearRect(0, 0, width, height)

        var cx = width / 2
        var cy = height / 2
        var r = Math.min(cx, cy) - 4

        /* Rings */
        ctx.strokeStyle = Qt.rgba(radarColor.r, radarColor.g, radarColor.b, 0.15)
        ctx.lineWidth = 1
        for (var ring = 1; ring <= 3; ring++) {
            ctx.beginPath()
            ctx.arc(cx, cy, r * ring / 3, 0, Math.PI * 2)
            ctx.stroke()
        }

        /* Cross lines */
        ctx.beginPath()
        ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy)
        ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r)
        ctx.strokeStyle = Qt.rgba(radarColor.r, radarColor.g, radarColor.b, 0.1)
        ctx.stroke()

        /* Sweep */
        var sweepRad = sweepAngle * Math.PI / 180
        var gradient = ctx.createConicalGradient(cx, cy, -sweepRad)
        gradient.addColorStop(0, Qt.rgba(radarColor.r, radarColor.g, radarColor.b, 0.3))
        gradient.addColorStop(0.1, Qt.rgba(radarColor.r, radarColor.g, radarColor.b, 0.05))
        gradient.addColorStop(0.5, "transparent")
        gradient.addColorStop(1, "transparent")

        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.arc(cx, cy, r, -sweepRad, -sweepRad + Math.PI * 0.5)
        ctx.closePath()
        ctx.fillStyle = gradient
        ctx.fill()

        /* Sweep line */
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(cx + r * Math.cos(-sweepRad), cy + r * Math.sin(-sweepRad))
        ctx.strokeStyle = radarColor
        ctx.lineWidth = 2
        ctx.stroke()

        /* Targets */
        for (var i = 0; i < targets.length; i++) {
            var t = targets[i]
            var tx = cx + t.x * r
            var ty = cy + t.y * r

            ctx.beginPath()
            ctx.arc(tx, ty, 3, 0, Math.PI * 2)
            ctx.fillStyle = t.color || radarColor
            ctx.fill()

            /* Glow */
            ctx.beginPath()
            ctx.arc(tx, ty, 6, 0, Math.PI * 2)
            ctx.fillStyle = Qt.rgba(radarColor.r, radarColor.g, radarColor.b, 0.2)
            ctx.fill()
        }

        /* Center dot */
        ctx.beginPath()
        ctx.arc(cx, cy, 3, 0, Math.PI * 2)
        ctx.fillStyle = radarColor
        ctx.fill()
    }
}
