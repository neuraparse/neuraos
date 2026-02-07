import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: taskbar
    height: Theme.taskbarH
    color: Theme.taskbarBg
    radius: 20
    border.width: 1
    border.color: Theme.glassBorder

    property var runningWindows: []
    property bool startMenuOpen: false

    signal startToggled()
    signal notificationToggled()
    signal windowClicked(int windowId)
    signal windowRightClicked(int windowId, real globalX, real globalY)

    Behavior on color { ColorAnimation { duration: Theme.animNormal } }

    /* Subtle top highlight */
    Rectangle {
        anchors.top: parent.top
        anchors.topMargin: 1
        anchors.left: parent.left
        anchors.leftMargin: 20
        anchors.right: parent.right
        anchors.rightMargin: 20
        height: 1
        color: Qt.rgba(1, 1, 1, 0.04)
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 10
        anchors.rightMargin: 14
        spacing: 0

        /* ─── Start Button ─── */
        Rectangle {
            width: 44; height: 44
            Layout.alignment: Qt.AlignVCenter
            radius: 12
            color: startBtnMa.containsMouse
                   ? Theme.taskbarHover
                   : startMenuOpen ? Theme.taskbarActive : "transparent"

            Behavior on color { ColorAnimation { duration: 80 } }

            Canvas {
                anchors.centerIn: parent
                width: 22; height: 22
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, 22, 22)
                    var clr = startMenuOpen ? Theme.primary
                              : startBtnMa.containsMouse ? Theme.text
                              : Theme.textDim
                    ctx.fillStyle = clr
                    ctx.strokeStyle = clr
                    ctx.lineWidth = 1.4
                    ctx.lineCap = "round"
                    /* Neural network hub: center + 6 outer nodes */
                    var cx = 11, cy = 11, r = 8
                    var nodes = []
                    for (var i = 0; i < 6; i++) {
                        var a = i * Math.PI / 3 - Math.PI / 2
                        nodes.push([cx + r * Math.cos(a), cy + r * Math.sin(a)])
                    }
                    /* Connection lines from center to outer */
                    for (var j = 0; j < 6; j++) {
                        ctx.beginPath()
                        ctx.moveTo(cx, cy)
                        ctx.lineTo(nodes[j][0], nodes[j][1])
                        ctx.stroke()
                    }
                    /* Outer ring connections */
                    for (var k = 0; k < 6; k++) {
                        var next = (k + 1) % 6
                        ctx.beginPath()
                        ctx.moveTo(nodes[k][0], nodes[k][1])
                        ctx.lineTo(nodes[next][0], nodes[next][1])
                        ctx.stroke()
                    }
                    /* Outer nodes */
                    for (var m = 0; m < 6; m++) {
                        ctx.beginPath()
                        ctx.arc(nodes[m][0], nodes[m][1], 2.0, 0, Math.PI * 2)
                        ctx.fill()
                    }
                    /* Center node (larger) */
                    ctx.beginPath()
                    ctx.arc(cx, cy, 2.8, 0, Math.PI * 2)
                    ctx.fill()
                }
                property bool _sm: startMenuOpen
                property bool _hv: startBtnMa.containsMouse
                on_SmChanged: requestPaint()
                on_HvChanged: requestPaint()
                Component.onCompleted: requestPaint()
            }

            MouseArea {
                id: startBtnMa
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: taskbar.startToggled()
            }
        }

        /* ─── Separator ─── */
        Rectangle {
            width: 1; height: 24
            Layout.alignment: Qt.AlignVCenter
            Layout.leftMargin: 6; Layout.rightMargin: 6
            color: Theme.surfaceLight
        }

        /* ─── Running Windows ─── */
        Row {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignVCenter
            spacing: 4

            Repeater {
                model: runningWindows.length

                Rectangle {
                    property var winData: runningWindows[index]

                    width: 44; height: 44
                    anchors.verticalCenter: parent.verticalCenter
                    radius: Theme.radiusTiny
                    color: winItemMa.containsMouse
                           ? Theme.taskbarHover
                           : (winData && winData.focused) ? Theme.taskbarActive
                           : "transparent"

                    opacity: 0
                    scale: 0.85

                    Component.onCompleted: {
                        opacity = 1.0
                        scale = 1.0
                    }

                    Behavior on opacity {
                        NumberAnimation { duration: 250; easing.type: Easing.OutCubic }
                    }
                    Behavior on scale {
                        NumberAnimation { duration: 250; easing.type: Easing.OutCubic }
                    }
                    Behavior on color { ColorAnimation { duration: 80 } }

                    Components.CanvasIcon {
                        anchors.centerIn: parent
                        iconName: winData ? winData.icon : ""
                        iconColor: winData ? winData.color : Theme.text
                        iconSize: 18
                    }

                    /* Bottom indicator */
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 2
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: winData && !winData.minimized ? 16 : 6
                        height: 3
                        radius: 1.5
                        color: winData ? winData.color : Theme.primary
                        Behavior on width {
                            NumberAnimation { duration: Theme.animFast; easing.type: Easing.OutCubic }
                        }
                    }

                    MouseArea {
                        id: winItemMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        acceptedButtons: Qt.LeftButton | Qt.RightButton

                        property bool isHovered: containsMouse

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

                    /* Hover scale effect */
                    transform: Scale {
                        origin.x: 22; origin.y: 22
                        xScale: winItemMa.isHovered ? 1.06 : 1.0
                        yScale: winItemMa.isHovered ? 1.06 : 1.0
                        Behavior on xScale {
                            NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
                        }
                        Behavior on yScale {
                            NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
                        }
                    }
                }
            }
        }

        /* ─── System Tray ─── */
        Row {
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignVCenter
            spacing: 14

            /* CPU indicator */
            Row {
                spacing: 5
                anchors.verticalCenter: parent.verticalCenter
                Rectangle {
                    width: 5; height: 5; radius: 2.5
                    anchors.verticalCenter: parent.verticalCenter
                    color: SystemInfo.cpuUsage > 80 ? Theme.error
                           : SystemInfo.cpuUsage > 50 ? Theme.warning
                           : Theme.success
                }
                Text {
                    text: Math.round(SystemInfo.cpuUsage) + "%"
                    font.pixelSize: 11
                    font.family: Theme.fontFamily
                    color: Theme.textDim
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            /* Memory indicator */
            Row {
                spacing: 5
                anchors.verticalCenter: parent.verticalCenter
                Rectangle {
                    width: 5; height: 5; radius: 2.5
                    anchors.verticalCenter: parent.verticalCenter
                    color: {
                        var memPct = SystemInfo.memoryTotal > 0
                                     ? SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100
                                     : 0
                        return memPct > 80 ? Theme.error
                               : memPct > 60 ? Theme.warning
                               : Theme.success
                    }
                }
                Text {
                    text: (SystemInfo.memoryUsed / 1048576).toFixed(0) + "M"
                    font.pixelSize: 11
                    font.family: Theme.fontFamily
                    color: Theme.textDim
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            /* NPU badge */
            Text {
                text: "NPU"
                font.pixelSize: 10
                font.bold: true
                font.family: Theme.fontFamily
                color: Theme.primary
                visible: NPUMonitor.deviceCount > 0
                anchors.verticalCenter: parent.verticalCenter
            }

            /* WiFi icon */
            Components.CanvasIcon {
                anchors.verticalCenter: parent.verticalCenter
                iconName: "wifi"
                iconSize: 14
                iconColor: Theme.success
            }

            /* Separator */
            Rectangle {
                width: 1; height: 24
                color: Theme.surfaceLight
                anchors.verticalCenter: parent.verticalCenter
            }

            /* Notification bell */
            Rectangle {
                width: 32; height: 32
                radius: Theme.radiusSmall
                anchors.verticalCenter: parent.verticalCenter
                color: notifMa.containsMouse ? Theme.taskbarHover : "transparent"

                Behavior on color { ColorAnimation { duration: 80 } }

                Components.CanvasIcon {
                    anchors.centerIn: parent
                    iconName: "bell"
                    iconColor: Theme.textDim
                    iconSize: 15
                }

                MouseArea {
                    id: notifMa
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: taskbar.notificationToggled()
                }
            }

            /* Clock */
            Column {
                anchors.verticalCenter: parent.verticalCenter
                spacing: -2

                Text {
                    id: clockText
                    font.pixelSize: 13
                    font.bold: true
                    font.family: Theme.fontFamily
                    color: Theme.text

                    Timer {
                        interval: 1000
                        running: true
                        repeat: true
                        triggeredOnStart: true
                        onTriggered: {
                            var d = new Date()
                            clockText.text = Qt.formatTime(d, "HH:mm")
                            dateText.text = Qt.formatDate(d, "dd MMM")
                        }
                    }
                }

                Text {
                    id: dateText
                    font.pixelSize: 9
                    font.family: Theme.fontFamily
                    color: Theme.textDim
                }
            }
        }
    }
}
