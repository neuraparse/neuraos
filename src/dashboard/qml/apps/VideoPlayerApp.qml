import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: videoApp
    anchors.fill: parent

    property bool isPlaying: false
    property int currentVideo: 0
    property real currentTime: 0
    property real volume: 0.75

    property var playlist: [
        { title: "Neural Architecture Deep Dive",     author: "AI Academy",      views: "1.2M", duration: 600, color: "#667eea" },
        { title: "Quantum Computing Explained",       author: "Qubit Labs",      views: "843K", duration: 480, color: "#f5576c" },
        { title: "Building OS Kernels from Scratch",  author: "Systems Forge",   views: "2.1M", duration: 720, color: "#43e97b" },
        { title: "Cybersecurity in 2026",             author: "DefenseNet",      views: "567K", duration: 540, color: "#fbbf24" },
        { title: "Rust for Systems Programming",      author: "Code Foundry",    views: "1.8M", duration: 660, color: "#a78bfa" },
        { title: "Edge AI Deployment Guide",          author: "NPU Workshop",    views: "392K", duration: 420, color: "#38bdf8" }
    ]

    Timer {
        id: playbackTimer
        interval: 250; running: isPlaying; repeat: true
        onTriggered: {
            if (currentTime < playlist[currentVideo].duration)
                currentTime += 0.25
            else { isPlaying = false; currentTime = playlist[currentVideo].duration }
        }
    }

    function fmtTime(secs) {
        var m = Math.floor(secs / 60)
        var s = Math.floor(secs % 60)
        return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Main Player Area ─── */
            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                spacing: 0

                /* Video viewport */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: "#0A0A0F"

                    /* Mock video gradient */
                    Rectangle {
                        anchors.fill: parent; anchors.margins: 0
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: Qt.rgba(0.05, 0.05, 0.08, 1) }
                            GradientStop {
                                position: 1.0
                                color: {
                                    var c = playlist[currentVideo].color
                                    var r = parseInt(c.substring(1,3), 16) / 255
                                    var g = parseInt(c.substring(3,5), 16) / 255
                                    var b = parseInt(c.substring(5,7), 16) / 255
                                    return Qt.rgba(r, g, b, 0.2)
                                }
                            }
                        }
                    }

                    /* Central play icon overlay */
                    Rectangle {
                        anchors.centerIn: parent
                        width: 72; height: 72; radius: 36
                        color: Qt.rgba(0, 0, 0, 0.55)
                        visible: !isPlaying
                        border.width: 2; border.color: Qt.rgba(1, 1, 1, 0.2)

                        Components.CanvasIcon {
                            anchors.centerIn: parent; anchors.horizontalCenterOffset: 2
                            iconName: "play"; iconSize: 32; iconColor: "#FFFFFF"
                        }

                        MouseArea {
                            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                            onClicked: isPlaying = true
                        }
                    }

                    /* Click anywhere to toggle */
                    MouseArea {
                        anchors.fill: parent; z: -1
                        onClicked: isPlaying = !isPlaying
                    }

                    /* Video title overlay */
                    Text {
                        anchors.top: parent.top; anchors.left: parent.left
                        anchors.margins: 16
                        text: playlist[currentVideo].title
                        font.pixelSize: 16; font.weight: Font.DemiBold
                        font.family: Theme.fontFamily; color: "#FFFFFF"
                        opacity: isPlaying ? 0.0 : 0.9
                        Behavior on opacity { NumberAnimation { duration: Theme.animFast } }
                    }
                }

                /* ─── Transport controls bar ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 80
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; spacing: 0

                        /* Progress bar */
                        Item {
                            Layout.fillWidth: true; Layout.preferredHeight: 20
                            Layout.leftMargin: 12; Layout.rightMargin: 12

                            Rectangle {
                                id: progressTrack
                                anchors.left: parent.left; anchors.right: parent.right
                                anchors.verticalCenter: parent.verticalCenter
                                height: 4; radius: 2
                                color: Theme.surfaceLight

                                Rectangle {
                                    width: playlist[currentVideo].duration > 0
                                         ? parent.width * (currentTime / playlist[currentVideo].duration) : 0
                                    height: parent.height; radius: 2
                                    color: Theme.primary
                                    Behavior on width { NumberAnimation { duration: 150 } }
                                }

                                Rectangle {
                                    x: playlist[currentVideo].duration > 0
                                     ? parent.width * (currentTime / playlist[currentVideo].duration) - 6 : -6
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: 12; height: 12; radius: 6
                                    color: Theme.primary
                                    visible: progressMa.containsMouse || progressMa.pressed
                                }
                            }

                            MouseArea {
                                id: progressMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    var ratio = Math.max(0, Math.min(1, mouse.x / progressTrack.width))
                                    currentTime = ratio * playlist[currentVideo].duration
                                }
                            }
                        }

                        /* Controls row */
                        RowLayout {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            Layout.leftMargin: 12; Layout.rightMargin: 12
                            spacing: 8

                            /* Previous */
                            Rectangle {
                                width: 34; height: 34; radius: 17
                                color: prevVMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Components.CanvasIcon { anchors.centerIn: parent; iconName: "skip-back"; iconSize: 18; iconColor: Theme.text }
                                MouseArea {
                                    id: prevVMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: { if (currentVideo > 0) { currentVideo--; currentTime = 0 } }
                                }
                            }

                            /* Play/Pause */
                            Rectangle {
                                width: 42; height: 42; radius: 21
                                color: playVMa.pressed ? Qt.darker(Theme.primary, 1.1) : Theme.primary
                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: isPlaying ? "pause" : "play"
                                    iconSize: 22; iconColor: "#FFFFFF"
                                }
                                MouseArea {
                                    id: playVMa; anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: isPlaying = !isPlaying
                                }
                            }

                            /* Next */
                            Rectangle {
                                width: 34; height: 34; radius: 17
                                color: nextVMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Components.CanvasIcon { anchors.centerIn: parent; iconName: "skip-forward"; iconSize: 18; iconColor: Theme.text }
                                MouseArea {
                                    id: nextVMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: { if (currentVideo < playlist.length - 1) { currentVideo++; currentTime = 0 } }
                                }
                            }

                            /* Time label */
                            Text {
                                text: fmtTime(currentTime) + " / " + fmtTime(playlist[currentVideo].duration)
                                font.pixelSize: 12; font.family: Theme.fontFamily
                                color: Theme.textDim
                            }

                            Item { Layout.fillWidth: true }

                            /* Volume icon */
                            Components.CanvasIcon {
                                iconName: volume > 0 ? "volume" : "minus"
                                iconSize: 16; iconColor: Theme.textDim
                            }

                            /* Volume slider */
                            Rectangle {
                                width: 80; height: 4; radius: 2
                                color: Theme.surfaceLight
                                Layout.alignment: Qt.AlignVCenter

                                Rectangle {
                                    width: parent.width * volume
                                    height: parent.height; radius: 2
                                    color: Theme.secondary
                                }

                                Rectangle {
                                    x: parent.width * volume - 5
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: 10; height: 10; radius: 5
                                    color: Theme.secondary
                                    visible: volMa.containsMouse || volMa.pressed
                                }

                                MouseArea {
                                    id: volMa; anchors.fill: parent
                                    anchors.topMargin: -6; anchors.bottomMargin: -6
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: volume = Math.max(0, Math.min(1, mouse.x / parent.width))
                                }
                            }

                            /* Fullscreen */
                            Rectangle {
                                width: 34; height: 34; radius: 17
                                color: fsMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Components.CanvasIcon { anchors.centerIn: parent; iconName: "fullscreen"; iconSize: 16; iconColor: Theme.textDim }
                                MouseArea { id: fsMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                            }
                        }
                    }
                }

                /* ─── Video Info ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 70
                    color: Theme.surfaceAlt

                    RowLayout {
                        anchors.fill: parent; anchors.margins: 14; spacing: 12

                        ColumnLayout {
                            Layout.fillWidth: true; spacing: 3

                            Text {
                                text: playlist[currentVideo].title
                                font.pixelSize: 15; font.weight: Font.DemiBold
                                font.family: Theme.fontFamily; color: Theme.text
                                elide: Text.ElideRight; Layout.fillWidth: true
                            }

                            RowLayout {
                                spacing: 12
                                Text {
                                    text: playlist[currentVideo].author
                                    font.pixelSize: 12; font.family: Theme.fontFamily; color: Theme.primary
                                }
                                Text {
                                    text: playlist[currentVideo].views + " views"
                                    font.pixelSize: 11; font.family: Theme.fontFamily; color: Theme.textMuted
                                }
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

            /* ─── Playlist Sidebar ─── */
            Rectangle {
                Layout.preferredWidth: 220; Layout.fillHeight: true
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent; spacing: 0

                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 40
                        color: "transparent"

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; spacing: 6

                            Components.CanvasIcon { iconName: "list"; iconSize: 14; iconColor: Theme.primary }

                            Text {
                                Layout.fillWidth: true; text: "Playlist"
                                font.pixelSize: 13; font.weight: Font.DemiBold
                                font.family: Theme.fontFamily; color: Theme.text
                            }

                            Text {
                                text: playlist.length + " videos"
                                font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textMuted
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                    Flickable {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        contentHeight: plCol.height; clip: true

                        Column {
                            id: plCol; width: parent.width

                            Repeater {
                                model: playlist.length

                                Rectangle {
                                    width: parent.width; height: 72
                                    color: index === currentVideo ? Theme.surfaceAlt
                                         : plItemMa.containsMouse ? Qt.rgba(Theme.surfaceAlt.r, Theme.surfaceAlt.g, Theme.surfaceAlt.b, 0.5)
                                         : "transparent"

                                    RowLayout {
                                        anchors.fill: parent; anchors.margins: 8; spacing: 10

                                        /* Thumbnail (colored rectangle) */
                                        Rectangle {
                                            Layout.preferredWidth: 80; Layout.preferredHeight: 50
                                            radius: Theme.radiusTiny
                                            color: playlist[index].color

                                            /* Duration badge */
                                            Rectangle {
                                                anchors.right: parent.right; anchors.bottom: parent.bottom
                                                anchors.margins: 3
                                                width: durLbl.implicitWidth + 6; height: 14; radius: 3
                                                color: Qt.rgba(0, 0, 0, 0.7)

                                                Text {
                                                    id: durLbl; anchors.centerIn: parent
                                                    text: fmtTime(playlist[index].duration)
                                                    font.pixelSize: 9; color: "#FFFFFF"
                                                }
                                            }

                                            /* Play indicator */
                                            Components.CanvasIcon {
                                                anchors.centerIn: parent
                                                iconName: "play"; iconSize: 18
                                                iconColor: "#FFFFFF"
                                                visible: index === currentVideo && isPlaying
                                                opacity: 0.8
                                            }
                                        }

                                        ColumnLayout {
                                            Layout.fillWidth: true; spacing: 3

                                            Text {
                                                Layout.fillWidth: true
                                                text: playlist[index].title
                                                font.pixelSize: 11; font.weight: Font.DemiBold
                                                font.family: Theme.fontFamily
                                                color: index === currentVideo ? Theme.primary : Theme.text
                                                elide: Text.ElideRight; maximumLineCount: 2
                                                wrapMode: Text.WordWrap
                                            }

                                            Text {
                                                text: playlist[index].author
                                                font.pixelSize: 10; font.family: Theme.fontFamily
                                                color: Theme.textMuted; elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }

                                    MouseArea {
                                        id: plItemMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: { currentVideo = index; currentTime = 0; isPlaying = true }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
