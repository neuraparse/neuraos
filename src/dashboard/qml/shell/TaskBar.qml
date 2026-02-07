import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: taskbar
    radius: 20
    color: Theme.taskbarBg
    border.width: 1
    border.color: Theme.glassBorder

    property var runningWindows: []
    property bool startMenuOpen: false

    signal startToggled()
    signal notificationToggled()
    signal windowClicked(int windowId)
    signal windowRightClicked(int windowId, real globalX, real globalY)

    Behavior on color { ColorAnimation { duration: Theme.animNormal } }

    /* Top highlight (glassmorphic) */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 1; anchors.leftMargin: 12; anchors.rightMargin: 12
        height: 1
        color: Qt.rgba(1, 1, 1, Theme.darkMode ? 0.06 : 0.25)
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 6
        anchors.rightMargin: 12
        spacing: 0

        /* Start button */
        Rectangle {
            width: 42; height: 36
            Layout.alignment: Qt.AlignVCenter
            radius: Theme.radiusSmall
            color: startBtnMa.containsMouse || startMenuOpen ? Theme.taskbarActive : "transparent"

            Canvas {
                anchors.centerIn: parent
                width: 20; height: 20
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, 20, 20)
                    var clr = startMenuOpen ? Theme.primary : Theme.textDim
                    ctx.fillStyle = clr
                    ctx.strokeStyle = clr
                    ctx.lineWidth = 1.4
                    ctx.lineCap = "round"
                    /* Neural network hub: center + 6 outer nodes */
                    var cx = 10, cy = 10, r = 7
                    var nodes = []
                    for (var i = 0; i < 6; i++) {
                        var a = i * Math.PI / 3 - Math.PI / 2
                        nodes.push([cx + r * Math.cos(a), cy + r * Math.sin(a)])
                    }
                    /* Connection lines */
                    for (var j = 0; j < 6; j++) {
                        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nodes[j][0], nodes[j][1]); ctx.stroke()
                    }
                    /* Outer ring connections */
                    for (var k = 0; k < 6; k++) {
                        var next = (k + 1) % 6
                        ctx.beginPath(); ctx.moveTo(nodes[k][0], nodes[k][1]); ctx.lineTo(nodes[next][0], nodes[next][1]); ctx.stroke()
                    }
                    /* Outer nodes */
                    for (var m = 0; m < 6; m++) {
                        ctx.beginPath(); ctx.arc(nodes[m][0], nodes[m][1], 1.8, 0, Math.PI * 2); ctx.fill()
                    }
                    /* Center node (larger) */
                    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill()
                }
                property bool _sm: startMenuOpen
                on_SmChanged: requestPaint()
                Component.onCompleted: requestPaint()
            }

            MouseArea {
                id: startBtnMa; anchors.fill: parent
                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                onClicked: taskbar.startToggled()
            }
            Behavior on color { ColorAnimation { duration: Theme.animFast } }
        }

        Rectangle {
            width: 1; height: 22
            Layout.alignment: Qt.AlignVCenter
            color: Theme.surfaceLight
            Layout.leftMargin: 4; Layout.rightMargin: 4
        }

        /* Running windows */
        Row {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignVCenter
            spacing: 2

            Repeater {
                model: runningWindows.length

                Rectangle {
                    property var winData: runningWindows[index]
                    property bool showText: taskbar.width > 700
                    width: {
                        var count = runningWindows.length
                        if (count <= 0) return 48
                        var avail = taskbar.width - 280
                        var maxW = showText ? 180 : 48
                        var perItem = Math.floor(avail / count) - 2
                        return Math.max(42, Math.min(maxW, perItem))
                    }
                    height: 36
                    anchors.verticalCenter: parent.verticalCenter
                    radius: Theme.radiusSmall
                    color: winItemMa.containsMouse ? Theme.taskbarHover :
                           (winData && winData.focused) ? Theme.taskbarActive : "transparent"
                    opacity: 0; scale: 0.8

                    Component.onCompleted: { opacity = 1.0; scale = 1.0 }
                    Behavior on opacity { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }
                    Behavior on scale { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 8
                        anchors.rightMargin: 8
                        spacing: 6

                        Components.CanvasIcon {
                            Layout.alignment: Qt.AlignVCenter
                            iconName: winData ? winData.icon : ""
                            iconColor: winData ? winData.color : Theme.text
                            iconSize: 16
                        }
                        Text {
                            Layout.fillWidth: true
                            text: winData ? winData.title : ""
                            font.pixelSize: 11; font.family: Theme.fontFamily
                            color: Theme.text
                            visible: showText && parent.width > 60
                            elide: Text.ElideRight
                        }
                    }

                    Rectangle {
                        anchors.bottom: parent.bottom; anchors.bottomMargin: 1
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: winData && !winData.minimized ? 16 : 4
                        height: 2; radius: 1
                        color: winData ? winData.color : Theme.primary
                        Behavior on width { NumberAnimation { duration: Theme.animFast } }
                    }

                    MouseArea {
                        id: winItemMa; anchors.fill: parent
                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        acceptedButtons: Qt.LeftButton | Qt.RightButton
                        onClicked: {
                            if (!winData) return
                            if (mouse.button === Qt.RightButton) {
                                var gp = mapToItem(null, mouse.x, mouse.y)
                                taskbar.windowRightClicked(winData.id, gp.x, gp.y)
                            } else {
                                taskbar.windowClicked(winData.id)
                            }
                        }
                    }
                    Behavior on color { ColorAnimation { duration: Theme.animFast } }
                }
            }
        }

        /* System tray */
        Row {
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignVCenter
            spacing: 12

            Row {
                spacing: 4; anchors.verticalCenter: parent.verticalCenter
                Rectangle {
                    width: 5; height: 5; radius: 3; anchors.verticalCenter: parent.verticalCenter
                    color: SystemInfo.cpuUsage > 80 ? Theme.error :
                           SystemInfo.cpuUsage > 50 ? Theme.warning : Theme.success
                }
                Text {
                    text: Math.round(SystemInfo.cpuUsage) + "%"
                    font.pixelSize: 11; color: Theme.textDim
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            Row {
                spacing: 4; anchors.verticalCenter: parent.verticalCenter
                Rectangle {
                    width: 5; height: 5; radius: 3; anchors.verticalCenter: parent.verticalCenter
                    color: {
                        var memPct = SystemInfo.memoryTotal > 0 ? SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
                        return memPct > 80 ? Theme.error : memPct > 60 ? Theme.warning : Theme.success
                    }
                }
                Text {
                    text: (SystemInfo.memoryUsed / 1048576).toFixed(0) + "M"
                    font.pixelSize: 11; color: Theme.textDim
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: "NPU"; font.pixelSize: 10; font.bold: true; color: Theme.primary
                visible: NPUMonitor.deviceCount > 0
                anchors.verticalCenter: parent.verticalCenter
            }

            Components.CanvasIcon {
                anchors.verticalCenter: parent.verticalCenter
                iconName: "wifi"; iconSize: 13; iconColor: Theme.success
            }

            Rectangle { width: 1; height: 18; color: Theme.surfaceLight; anchors.verticalCenter: parent.verticalCenter }

            Rectangle {
                width: 30; height: 30; radius: Theme.radiusSmall
                anchors.verticalCenter: parent.verticalCenter
                color: notifMa.containsMouse ? Theme.taskbarHover : "transparent"
                Components.CanvasIcon {
                    anchors.centerIn: parent; iconName: "bell"; iconColor: Theme.textDim; iconSize: 15
                }
                MouseArea {
                    id: notifMa; anchors.fill: parent
                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: taskbar.notificationToggled()
                }
            }

            Column {
                anchors.verticalCenter: parent.verticalCenter; spacing: -2
                Text {
                    id: clockText; font.pixelSize: 12; font.bold: true; color: Theme.text
                    Timer {
                        interval: 1000; running: true; repeat: true; triggeredOnStart: true
                        onTriggered: {
                            var d = new Date()
                            clockText.text = Qt.formatTime(d, "HH:mm")
                            dateText.text = Qt.formatDate(d, "dd MMM")
                        }
                    }
                }
                Text { id: dateText; font.pixelSize: 9; color: Theme.textDim }
            }
        }
    }
}
