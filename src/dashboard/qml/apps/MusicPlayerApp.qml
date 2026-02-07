import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: musicApp
    anchors.fill: parent

    property int currentTrack: 0
    property bool isPlaying: false
    property bool shuffleOn: false
    property int repeatMode: 0  /* 0=off, 1=all, 2=one */
    property real progress: 0.0
    property var playlist: [
        { title: "Neural Synthesis", artist: "AI Orchestra", album: "Digital Dreams", duration: "4:23" },
        { title: "Quantum Bits", artist: "Electron Wave", album: "Quantum State", duration: "3:45" },
        { title: "Silicon Dreams", artist: "NeuralBeat", album: "Circuit Board", duration: "5:12" },
        { title: "Deep Learning Blues", artist: "Tensor Flow", album: "Gradient Descent", duration: "3:56" },
        { title: "Binary Sunset", artist: "Code Runner", album: "Compiled", duration: "4:08" },
        { title: "Recursive Loop", artist: "Stack Overflow", album: "Infinite", duration: "3:30" },
        { title: "Kernel Panic", artist: "System Core", album: "Root Access", duration: "4:47" },
        { title: "Async Await", artist: "Promise Chain", album: "Event Loop", duration: "3:18" },
        { title: "Memory Leak", artist: "Garbage Collector", album: "Heap Space", duration: "5:01" },
        { title: "Firewall Waltz", artist: "Port Scanner", album: "Network Suite", duration: "4:33" }
    ]

    Timer {
        id: playTimer
        interval: 100
        running: isPlaying
        repeat: true
        onTriggered: {
            progress += 0.004
            if (progress >= 1.0) {
                progress = 0
                if (currentTrack < playlist.length - 1) currentTrack++
                else if (repeatMode === 1) currentTrack = 0
                else isPlaying = false
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* Left Panel: Playlist */
            Rectangle {
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 0

                    /* Playlist Header */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 44
                        color: "transparent"

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 14
                            anchors.rightMargin: 14
                            spacing: 8

                            Components.CanvasIcon {
                                iconName: "volume"
                                iconSize: 16
                                iconColor: Theme.primary
                            }

                            Text {
                                Layout.fillWidth: true
                                text: "Playlist"
                                font.pixelSize: 15
                                font.weight: Font.DemiBold
                                color: Theme.text
                            }

                            Text {
                                text: playlist.length + " tracks"
                                font.pixelSize: 11
                                color: Theme.textDim
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                    /* Track List */
                    Flickable {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentHeight: trackCol.height
                        clip: true

                        Column {
                            id: trackCol
                            width: parent.width

                            Repeater {
                                model: playlist.length

                                Rectangle {
                                    width: parent.width
                                    height: 52
                                    color: index === currentTrack ? Theme.surfaceAlt
                                         : trackMa.containsMouse ? Qt.rgba(Theme.surfaceAlt.r, Theme.surfaceAlt.g, Theme.surfaceAlt.b, 0.5)
                                         : "transparent"

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 14
                                        anchors.rightMargin: 14
                                        spacing: 10

                                        /* Track Number / Playing Indicator */
                                        Text {
                                            Layout.preferredWidth: 20
                                            text: index === currentTrack && isPlaying ? "▶" : (index + 1).toString()
                                            font.pixelSize: 12
                                            color: index === currentTrack ? Theme.primary : Theme.textMuted
                                            horizontalAlignment: Text.AlignCenter
                                        }

                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 2

                                            Text {
                                                Layout.fillWidth: true
                                                text: playlist[index].title
                                                font.pixelSize: 13
                                                font.weight: index === currentTrack ? Font.DemiBold : Font.Normal
                                                color: index === currentTrack ? Theme.primary : Theme.text
                                                elide: Text.ElideRight
                                            }

                                            Text {
                                                Layout.fillWidth: true
                                                text: playlist[index].artist
                                                font.pixelSize: 11
                                                color: Theme.textDim
                                                elide: Text.ElideRight
                                            }
                                        }

                                        Text {
                                            text: playlist[index].duration
                                            font.pixelSize: 11
                                            color: Theme.textMuted
                                        }
                                    }

                                    MouseArea {
                                        id: trackMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            currentTrack = index
                                            progress = 0
                                            isPlaying = true
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 1
                Layout.fillHeight: true
                color: Theme.surfaceLight
            }

            /* Right Panel: Now Playing */
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: Theme.background

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 24
                    spacing: 16

                    Item { Layout.fillHeight: true; Layout.preferredHeight: 10 }

                    /* Album Art (Canvas Generated) */
                    Rectangle {
                        Layout.alignment: Qt.AlignHCenter
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 200
                        radius: Theme.radius
                        color: Theme.surface

                        Canvas {
                            id: albumArt
                            anchors.fill: parent
                            anchors.margins: 1

                            property var colors: [
                                ["#667eea", "#764ba2"],
                                ["#f093fb", "#f5576c"],
                                ["#4facfe", "#00f2fe"],
                                ["#43e97b", "#38f9d7"],
                                ["#fa709a", "#fee140"],
                                ["#a18cd1", "#fbc2eb"],
                                ["#fccb90", "#d57eeb"],
                                ["#e0c3fc", "#8ec5fc"],
                                ["#f5576c", "#ff9a9e"],
                                ["#667eea", "#43e97b"]
                            ]

                            onPaint: {
                                var ctx = getContext("2d")
                                var w = width, h = height
                                var c = colors[currentTrack % colors.length]

                                var g = ctx.createLinearGradient(0, 0, w, h)
                                g.addColorStop(0, c[0])
                                g.addColorStop(1, c[1])
                                ctx.fillStyle = g
                                ctx.fillRect(0, 0, w, h)

                                /* Decorative circles */
                                ctx.globalAlpha = 0.15
                                for (var i = 0; i < 5; i++) {
                                    ctx.beginPath()
                                    var cx = w * (0.2 + Math.sin(i * 2.1) * 0.3)
                                    var cy = h * (0.3 + Math.cos(i * 1.7) * 0.3)
                                    var cr = 20 + i * 15
                                    ctx.arc(cx, cy, cr, 0, Math.PI * 2)
                                    ctx.fillStyle = "#FFFFFF"
                                    ctx.fill()
                                }
                                ctx.globalAlpha = 1.0

                                /* Music note symbol */
                                ctx.font = "48px serif"
                                ctx.fillStyle = "rgba(255,255,255,0.3)"
                                ctx.textAlign = "center"
                                ctx.fillText("♫", w / 2, h / 2 + 16)
                            }

                            Component.onCompleted: requestPaint()
                        }

                        Connections {
                            target: musicApp
                            function onCurrentTrackChanged() { albumArt.requestPaint() }
                        }
                    }

                    /* Track Info */
                    ColumnLayout {
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 4

                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: playlist[currentTrack].title
                            font.pixelSize: 20
                            font.weight: Font.DemiBold
                            color: Theme.text
                        }

                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: playlist[currentTrack].artist + " — " + playlist[currentTrack].album
                            font.pixelSize: 13
                            color: Theme.textDim
                        }
                    }

                    /* Progress Bar */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        spacing: 4

                        Rectangle {
                            Layout.fillWidth: true
                            height: 4
                            radius: 2
                            color: Theme.surfaceLight

                            Rectangle {
                                width: parent.width * progress
                                height: parent.height
                                radius: 2
                                color: Theme.primary

                                Behavior on width { NumberAnimation { duration: 100 } }
                            }

                            /* Seek handle */
                            Rectangle {
                                x: parent.width * progress - 6
                                y: -4
                                width: 12; height: 12
                                radius: 6
                                color: Theme.primary
                                visible: seekMa.containsMouse || seekMa.pressed
                            }

                            MouseArea {
                                id: seekMa
                                anchors.fill: parent
                                anchors.topMargin: -8
                                anchors.bottomMargin: -8
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: progress = Math.max(0, Math.min(1, mouse.x / parent.width))
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            Text {
                                text: {
                                    var totalSec = parseDuration(playlist[currentTrack].duration) * progress
                                    return formatTime(totalSec)
                                }
                                font.pixelSize: 11
                                color: Theme.textMuted
                            }

                            Item { Layout.fillWidth: true }

                            Text {
                                text: playlist[currentTrack].duration
                                font.pixelSize: 11
                                color: Theme.textMuted
                            }
                        }
                    }

                    /* Controls */
                    RowLayout {
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 20

                        /* Shuffle */
                        Rectangle {
                            width: 36; height: 36; radius: 18
                            color: shuffleOn ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15) : "transparent"

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "shuffle"
                                iconSize: 16
                                iconColor: shuffleOn ? Theme.primary : Theme.textDim
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: shuffleOn = !shuffleOn
                            }
                        }

                        /* Previous */
                        Rectangle {
                            width: 40; height: 40; radius: 20
                            color: prevMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "skip-back"
                                iconSize: 20
                                iconColor: Theme.text
                            }

                            MouseArea {
                                id: prevMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    if (currentTrack > 0) currentTrack--
                                    progress = 0
                                }
                            }
                        }

                        /* Play / Pause */
                        Rectangle {
                            width: 56; height: 56; radius: 28
                            color: playMa.pressed ? Qt.darker(Theme.primary, 1.1) : Theme.primary

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: isPlaying ? "pause" : "play"
                                iconSize: 24
                                iconColor: "#FFFFFF"
                            }

                            MouseArea {
                                id: playMa; anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: isPlaying = !isPlaying
                            }
                        }

                        /* Next */
                        Rectangle {
                            width: 40; height: 40; radius: 20
                            color: nextMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "skip-forward"
                                iconSize: 20
                                iconColor: Theme.text
                            }

                            MouseArea {
                                id: nextMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    if (currentTrack < playlist.length - 1) currentTrack++
                                    progress = 0
                                }
                            }
                        }

                        /* Repeat */
                        Rectangle {
                            width: 36; height: 36; radius: 18
                            color: repeatMode > 0 ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15) : "transparent"

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "repeat"
                                iconSize: 16
                                iconColor: repeatMode > 0 ? Theme.primary : Theme.textDim
                            }

                            Text {
                                visible: repeatMode === 2
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                text: "1"
                                font.pixelSize: 8
                                font.weight: Font.Bold
                                color: Theme.primary
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: repeatMode = (repeatMode + 1) % 3
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }

    function parseDuration(dur) {
        var parts = dur.split(":")
        return parseInt(parts[0]) * 60 + parseInt(parts[1])
    }

    function formatTime(seconds) {
        var m = Math.floor(seconds / 60)
        var s = Math.floor(seconds % 60)
        return m + ":" + (s < 10 ? "0" : "") + s
    }
}
