import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 200; height: 120
    widgetTitle: "Weather"

    RowLayout {
        anchors.fill: parent
        spacing: 12

        /* Weather icon (Canvas-drawn sun) */
        Canvas {
            width: 48; height: 48
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                /* Sun circle */
                ctx.beginPath()
                ctx.arc(24, 24, 10, 0, Math.PI * 2)
                ctx.fillStyle = "#FBBF24"
                ctx.fill()
                /* Rays */
                ctx.strokeStyle = "#FBBF24"
                ctx.lineWidth = 2
                ctx.lineCap = "round"
                for (var i = 0; i < 8; i++) {
                    var angle = (i * 45) * Math.PI / 180
                    ctx.beginPath()
                    ctx.moveTo(24 + Math.cos(angle) * 14, 24 + Math.sin(angle) * 14)
                    ctx.lineTo(24 + Math.cos(angle) * 19, 24 + Math.sin(angle) * 19)
                    ctx.stroke()
                }
            }
            Component.onCompleted: requestPaint()
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2

            Text {
                text: "72\u00B0F"
                font.pixelSize: 28
                font.weight: Font.Bold
                font.family: Theme.fontFamily
                color: Theme.text
            }

            Text {
                text: "Partly Cloudy"
                font.pixelSize: 11
                font.family: Theme.fontFamily
                color: Theme.textDim
            }

            Text {
                text: "San Francisco"
                font.pixelSize: 9
                font.family: Theme.fontFamily
                color: Theme.textMuted
            }

            RowLayout {
                spacing: 10
                Text { text: "H: 76\u00B0"; font.pixelSize: 9; color: Theme.error; font.family: Theme.fontFamily }
                Text { text: "L: 58\u00B0"; font.pixelSize: 9; color: Theme.accent; font.family: Theme.fontFamily }
            }
        }
    }
}
