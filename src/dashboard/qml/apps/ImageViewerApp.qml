import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: imgViewerApp
    anchors.fill: parent

    property real zoomLevel: 1.0
    property real panX: 0
    property real panY: 0
    property int rotation: 0
    property int selectedThumb: 0

    function fitToView() {
        zoomLevel = 1.0; panX = 0; panY = 0
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Toolbar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 38
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10
                    anchors.rightMargin: 10
                    spacing: 2

                    Repeater {
                        model: [
                            { icon: "zoom-in", tip: "Zoom In", act: "zin" },
                            { icon: "zoom-out", tip: "Zoom Out", act: "zout" },
                            { icon: "fullscreen", tip: "Fit to View", act: "fit" },
                            { icon: "separator" },
                            { icon: "refresh", tip: "Rotate", act: "rotate" },
                            { icon: "separator" },
                            { icon: "arrow-left", tip: "Previous", act: "prev" },
                            { icon: "arrow-right", tip: "Next", act: "next" }
                        ]

                        Rectangle {
                            Layout.preferredWidth: modelData.icon === "separator" ? 1 : 32
                            Layout.preferredHeight: modelData.icon === "separator" ? 20 : 28
                            Layout.alignment: Qt.AlignVCenter
                            radius: modelData.icon === "separator" ? 0 : Theme.radiusSmall
                            color: modelData.icon === "separator" ? Theme.surfaceLight
                                 : ivBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Components.CanvasIcon {
                                visible: modelData.icon !== "separator"
                                anchors.centerIn: parent
                                iconName: modelData.icon || ""
                                iconSize: 14
                                iconColor: Theme.textDim
                            }

                            MouseArea {
                                id: ivBtnMa; anchors.fill: parent
                                hoverEnabled: true
                                enabled: modelData.icon !== "separator"
                                cursorShape: modelData.icon !== "separator" ? Qt.PointingHandCursor : Qt.ArrowCursor
                                onClicked: {
                                    if (modelData.act === "zin") zoomLevel = Math.min(zoomLevel * 1.25, 5.0)
                                    else if (modelData.act === "zout") zoomLevel = Math.max(zoomLevel / 1.25, 0.2)
                                    else if (modelData.act === "fit") fitToView()
                                    else if (modelData.act === "rotate") rotation = (rotation + 90) % 360
                                    else if (modelData.act === "prev") selectedThumb = Math.max(0, selectedThumb - 1)
                                    else if (modelData.act === "next") selectedThumb = Math.min(5, selectedThumb + 1)
                                }
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    Text {
                        text: Math.round(zoomLevel * 100) + "%"
                        font.pixelSize: 12
                        color: Theme.textDim
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Image Canvas Area */
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: Theme.darkMode ? "#0D0D10" : "#E0E0E4"
                clip: true

                /* Generated Art Canvas */
                Canvas {
                    id: artCanvas
                    width: 600; height: 400
                    anchors.centerIn: parent
                    scale: zoomLevel
                    rotation: imgViewerApp.rotation

                    transform: Translate { x: panX; y: panY }

                    Behavior on scale { NumberAnimation { duration: 150 } }
                    Behavior on rotation { NumberAnimation { duration: 200 } }

                    property var artStyles: [
                        { name: "Sunset Mountains", bg1: "#1a0533", bg2: "#FF6B35", m1: "#2d1b69", m2: "#1a0533", sun: "#FF6B35" },
                        { name: "Ocean Waves", bg1: "#0a1628", bg2: "#1e3a5f", m1: "#1565C0", m2: "#0D47A1", sun: "#4FC3F7" },
                        { name: "Northern Lights", bg1: "#0a0a1a", bg2: "#1a0533", m1: "#00695C", m2: "#004D40", sun: "#69F0AE" },
                        { name: "Desert Dunes", bg1: "#2d1b00", bg2: "#FF8F00", m1: "#BF360C", m2: "#8D6E63", sun: "#FFD54F" },
                        { name: "Neon City", bg1: "#0a0a1a", bg2: "#1a0033", m1: "#4A148C", m2: "#311B92", sun: "#E040FB" },
                        { name: "Arctic Dawn", bg1: "#1a2332", bg2: "#37474F", m1: "#546E7A", m2: "#455A64", sun: "#80DEEA" }
                    ]

                    onPaint: {
                        var ctx = getContext("2d")
                        var s = artStyles[selectedThumb]
                        var w = width, h = height

                        /* Sky gradient */
                        var sky = ctx.createLinearGradient(0, 0, 0, h * 0.7)
                        sky.addColorStop(0, s.bg1)
                        sky.addColorStop(1, s.bg2)
                        ctx.fillStyle = sky
                        ctx.fillRect(0, 0, w, h)

                        /* Sun/Moon */
                        var sunX = w * 0.7, sunY = h * 0.25, sunR = 40
                        var sunGrad = ctx.createRadialGradient(sunX, sunY, 0, sunX, sunY, sunR * 2)
                        sunGrad.addColorStop(0, s.sun)
                        sunGrad.addColorStop(0.3, Qt.rgba(1, 1, 1, 0.1))
                        sunGrad.addColorStop(1, "transparent")
                        ctx.fillStyle = sunGrad
                        ctx.fillRect(sunX - sunR * 2, sunY - sunR * 2, sunR * 4, sunR * 4)

                        ctx.beginPath()
                        ctx.arc(sunX, sunY, sunR, 0, Math.PI * 2)
                        ctx.fillStyle = s.sun
                        ctx.fill()

                        /* Mountains back */
                        ctx.beginPath()
                        ctx.moveTo(0, h * 0.65)
                        for (var i = 0; i <= 10; i++) {
                            var mx = (w / 10) * i
                            var my = h * 0.65 - Math.sin(i * 0.8 + 1) * h * 0.15 - Math.sin(i * 1.5) * h * 0.05
                            if (i === 0) ctx.moveTo(mx, my)
                            else ctx.lineTo(mx, my)
                        }
                        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath()
                        ctx.fillStyle = s.m1; ctx.fill()

                        /* Mountains front */
                        ctx.beginPath()
                        for (var j = 0; j <= 10; j++) {
                            var fx = (w / 10) * j
                            var fy = h * 0.75 - Math.sin(j * 1.2 + 2) * h * 0.12 - Math.cos(j * 0.7) * h * 0.04
                            if (j === 0) ctx.moveTo(fx, fy)
                            else ctx.lineTo(fx, fy)
                        }
                        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath()
                        ctx.fillStyle = s.m2; ctx.fill()

                        /* Ground */
                        var ground = ctx.createLinearGradient(0, h * 0.85, 0, h)
                        ground.addColorStop(0, s.m2)
                        ground.addColorStop(1, s.bg1)
                        ctx.fillStyle = ground
                        ctx.fillRect(0, h * 0.85, w, h * 0.15)

                        /* Stars */
                        ctx.fillStyle = "rgba(255,255,255,0.6)"
                        for (var k = 0; k < 30; k++) {
                            var sx = (Math.sin(k * 127.1 + 311.7) * 0.5 + 0.5) * w
                            var sy = (Math.sin(k * 269.5 + 183.3) * 0.5 + 0.5) * h * 0.5
                            var sr = 0.5 + Math.sin(k * 43.7) * 0.5
                            ctx.beginPath()
                            ctx.arc(sx, sy, sr, 0, Math.PI * 2)
                            ctx.fill()
                        }
                    }

                    Component.onCompleted: requestPaint()
                }

                /* Pan handler */
                MouseArea {
                    anchors.fill: parent
                    property point lastPos
                    onPressed: lastPos = Qt.point(mouse.x, mouse.y)
                    onPositionChanged: {
                        if (pressed) {
                            panX += (mouse.x - lastPos.x) / zoomLevel
                            panY += (mouse.y - lastPos.y) / zoomLevel
                            lastPos = Qt.point(mouse.x, mouse.y)
                        }
                    }
                    onWheel: {
                        if (wheel.angleDelta.y > 0)
                            zoomLevel = Math.min(zoomLevel * 1.1, 5.0)
                        else
                            zoomLevel = Math.max(zoomLevel / 1.1, 0.2)
                    }
                }

                /* Image Info Overlay */
                Rectangle {
                    anchors.bottom: parent.bottom
                    anchors.left: parent.left
                    anchors.margins: 10
                    width: infoRow.width + 20
                    height: 28
                    radius: 14
                    color: Qt.rgba(0, 0, 0, 0.6)

                    Row {
                        id: infoRow
                        anchors.centerIn: parent
                        spacing: 8

                        Text {
                            text: artCanvas.artStyles[selectedThumb].name
                            font.pixelSize: 11
                            color: "#FFFFFF"
                        }
                        Text {
                            text: "•"
                            font.pixelSize: 11
                            color: "#888"
                        }
                        Text {
                            text: "600 × 400"
                            font.pixelSize: 11
                            color: "#AAAAAA"
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Thumbnail Strip */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 72
                color: Theme.surface

                Row {
                    anchors.centerIn: parent
                    spacing: 8

                    Repeater {
                        model: artCanvas.artStyles.length

                        Rectangle {
                            width: 56; height: 56
                            radius: 6
                            border.width: index === selectedThumb ? 2 : 1
                            border.color: index === selectedThumb ? Theme.primary : Theme.surfaceLight

                            Canvas {
                                anchors.fill: parent
                                anchors.margins: 2
                                onPaint: {
                                    var ctx = getContext("2d")
                                    var s = artCanvas.artStyles[index]
                                    var w = width, h = height
                                    var g = ctx.createLinearGradient(0, 0, w, h)
                                    g.addColorStop(0, s.bg1)
                                    g.addColorStop(0.5, s.m1)
                                    g.addColorStop(1, s.bg2)
                                    ctx.fillStyle = g
                                    ctx.fillRect(0, 0, w, h)

                                    ctx.beginPath()
                                    ctx.arc(w * 0.7, h * 0.3, 6, 0, Math.PI * 2)
                                    ctx.fillStyle = s.sun
                                    ctx.fill()
                                }
                                Component.onCompleted: requestPaint()
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    selectedThumb = index
                                    artCanvas.requestPaint()
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
