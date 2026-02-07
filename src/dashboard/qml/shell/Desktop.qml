import QtQuick 2.15
import ".."

Item {
    id: desktop

    /* ─── Aurora gradient wallpaper base ─── */
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: Theme.darkMode ? "#080810" : "#E4E8F4" }
            GradientStop { position: 0.35; color: Theme.darkMode ? "#0A0A14" : "#DEE2F0" }
            GradientStop { position: 0.65; color: Theme.darkMode ? "#0C0C16" : "#D8DCE8" }
            GradientStop { position: 1.0; color: Theme.darkMode ? "#0A0A12" : "#D2D6E2" }
        }
    }

    /* ─── Aurora blobs (animated radial gradients) ─── */
    Canvas {
        id: auroraCanvas
        anchors.fill: parent

        property real t: 0
        property bool dark: Theme.darkMode

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            var w = width, h = height

            /* Blob 1: Deep blue (top-center) */
            var x1 = w * 0.45 + Math.sin(t * 0.4) * w * 0.03
            var y1 = h * -0.05 + Math.cos(t * 0.3) * h * 0.02
            var r1 = w * 0.45
            var g1 = ctx.createRadialGradient(x1, y1, 0, x1, y1, r1)
            g1.addColorStop(0, dark ? Qt.rgba(0.12, 0.22, 0.55, 0.18) : Qt.rgba(0.40, 0.55, 0.90, 0.08))
            g1.addColorStop(1, "transparent")
            ctx.fillStyle = g1
            ctx.fillRect(0, 0, w, h * 0.7)

            /* Blob 2: Purple (right) */
            var x2 = w * 0.82 + Math.sin(t * 0.35 + 1.5) * w * 0.025
            var y2 = h * 0.35 + Math.cos(t * 0.28 + 0.8) * h * 0.02
            var r2 = w * 0.35
            var g2 = ctx.createRadialGradient(x2, y2, 0, x2, y2, r2)
            g2.addColorStop(0, dark ? Qt.rgba(0.30, 0.14, 0.50, 0.14) : Qt.rgba(0.55, 0.40, 0.85, 0.06))
            g2.addColorStop(1, "transparent")
            ctx.fillStyle = g2
            ctx.fillRect(w * 0.3, 0, w * 0.7, h)

            /* Blob 3: Cyan (bottom-left) */
            var x3 = w * 0.15 + Math.cos(t * 0.32 + 2.2) * w * 0.02
            var y3 = h * 0.85 + Math.sin(t * 0.26 + 1.0) * h * 0.015
            var r3 = w * 0.38
            var g3 = ctx.createRadialGradient(x3, y3, 0, x3, y3, r3)
            g3.addColorStop(0, dark ? Qt.rgba(0.05, 0.25, 0.42, 0.12) : Qt.rgba(0.20, 0.60, 0.80, 0.05))
            g3.addColorStop(1, "transparent")
            ctx.fillStyle = g3
            ctx.fillRect(0, h * 0.3, w * 0.7, h * 0.7)

            /* Blob 4: Pink-magenta (bottom-right, subtle) */
            var x4 = w * 0.72 + Math.sin(t * 0.38 + 3.0) * w * 0.02
            var y4 = h * 1.05 + Math.cos(t * 0.22) * h * 0.02
            var r4 = w * 0.3
            var g4 = ctx.createRadialGradient(x4, y4, 0, x4, y4, r4)
            g4.addColorStop(0, dark ? Qt.rgba(0.40, 0.10, 0.30, 0.08) : Qt.rgba(0.70, 0.30, 0.55, 0.04))
            g4.addColorStop(1, "transparent")
            ctx.fillStyle = g4
            ctx.fillRect(w * 0.3, h * 0.5, w * 0.7, h * 0.5)
        }

        Timer {
            interval: 60; running: true; repeat: true
            onTriggered: { auroraCanvas.t += 0.012; auroraCanvas.requestPaint() }
        }

        Component.onCompleted: requestPaint()
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
    }

    Connections {
        target: Theme
        function onDarkModeChanged() { auroraCanvas.requestPaint() }
    }

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onClicked: { }
    }
}
