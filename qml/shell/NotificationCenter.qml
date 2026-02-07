import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: notifCenter
    width: Theme.notifCenterW
    color: Theme.notifCenterBg
    radius: Theme.radiusSmall
    border.width: 1
    border.color: Theme.glassBorder

    signal closePanel()

    /* ── Toggle state properties ── */
    property bool wifiActive: true
    property bool bluetoothActive: false
    property bool npuActive: NPUMonitor.deviceCount > 0
    property bool dndActive: false
    property bool themeActive: true
    property bool airplaneActive: false

    /* ── Slider state properties ── */
    property real brightnessValue: typeof Settings !== "undefined" && Settings.brightness !== undefined ? Settings.brightness : 75
    property real volumeValue: typeof Settings !== "undefined" && Settings.volume !== undefined ? Settings.volume : 80

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 12

        /* ═══════════════════════════════════════════════
           1. HEADER
           ═══════════════════════════════════════════════ */
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "Control Center"
                color: Theme.text
                font.pixelSize: 14
                font.bold: true
                font.family: Theme.fontFamily
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                width: 30; height: 30; radius: 15
                color: closeBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"

                Behavior on color { ColorAnimation { duration: Theme.animFast } }

                Components.CanvasIcon {
                    anchors.centerIn: parent
                    iconName: "close"
                    iconColor: Theme.textDim
                    iconSize: 14
                }

                MouseArea {
                    id: closeBtnMa
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: notifCenter.closePanel()
                }
            }
        }

        /* ═══════════════════════════════════════════════
           2. QUICK TOGGLES
           ═══════════════════════════════════════════════ */
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: toggleGrid.implicitHeight + 20
            radius: 12
            color: Theme.glass
            border.width: 1
            border.color: Theme.glassBorder

            GridLayout {
                id: toggleGrid
                anchors.fill: parent
                anchors.margins: 10
                columns: 3
                rows: 2
                columnSpacing: 8
                rowSpacing: 8

                /* --- WiFi --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.wifiActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: "wifi"
                            iconSize: 16
                            iconColor: notifCenter.wifiActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "WiFi"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: notifCenter.wifiActive = !notifCenter.wifiActive
                    }
                }

                /* --- Bluetooth --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.bluetoothActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: "bluetooth"
                            iconSize: 16
                            iconColor: notifCenter.bluetoothActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "Bluetooth"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: notifCenter.bluetoothActive = !notifCenter.bluetoothActive
                    }
                }

                /* --- NPU --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.npuActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: "chip"
                            iconSize: 16
                            iconColor: notifCenter.npuActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "NPU"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: notifCenter.npuActive = !notifCenter.npuActive
                    }
                }

                /* --- Do Not Disturb --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.dndActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: "moon"
                            iconSize: 16
                            iconColor: notifCenter.dndActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "DND"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: notifCenter.dndActive = !notifCenter.dndActive
                    }
                }

                /* --- Dark / Light Theme --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.themeActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: Theme.darkMode ? "moon" : "sun"
                            iconSize: 16
                            iconColor: notifCenter.themeActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: Theme.darkMode ? "Dark" : "Light"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: Settings.theme = (Settings.theme === "dark" ? "light" : "dark")
                    }
                }

                /* --- Airplane Mode --- */
                Rectangle {
                    Layout.fillWidth: true
                    height: 56
                    radius: Theme.radiusTiny
                    color: notifCenter.airplaneActive
                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                        : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Column {
                        anchors.centerIn: parent
                        spacing: 4

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: "airplane"
                            iconSize: 16
                            iconColor: notifCenter.airplaneActive ? Theme.primary : Theme.textDim
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "Airplane"
                            font.pixelSize: 9
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: notifCenter.airplaneActive = !notifCenter.airplaneActive
                    }
                }
            }
        }

        /* ═══════════════════════════════════════════════
           3. BRIGHTNESS SLIDER
           ═══════════════════════════════════════════════ */
        Rectangle {
            Layout.fillWidth: true
            height: 44
            radius: Theme.radiusTiny
            color: Theme.surfaceAlt

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 10

                Components.CanvasIcon {
                    iconName: "brightness"
                    iconSize: 14
                    iconColor: Theme.textDim
                }

                /* Custom slider track + handle */
                Item {
                    Layout.fillWidth: true
                    height: 24

                    /* Track background */
                    Rectangle {
                        id: brightnessTrack
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width
                        height: 4
                        radius: 2
                        color: Theme.surfaceLight

                        /* Active fill */
                        Rectangle {
                            width: parent.width * (notifCenter.brightnessValue / 100)
                            height: parent.height
                            radius: 2
                            color: Theme.primary

                            Behavior on width { NumberAnimation { duration: 60 } }
                        }
                    }

                    /* Draggable handle */
                    Rectangle {
                        id: brightnessHandle
                        width: 16; height: 16; radius: 8
                        color: Theme.primary
                        x: (notifCenter.brightnessValue / 100) * (parent.width - 16)
                        anchors.verticalCenter: parent.verticalCenter

                        Behavior on x { NumberAnimation { duration: 60 } }
                    }

                    MouseArea {
                        anchors.fill: parent
                        onPressed: updateBrightness(mouse)
                        onPositionChanged: if (pressed) updateBrightness(mouse)
                        function updateBrightness(m) {
                            var v = Math.max(0, Math.min(100, (m.x / width) * 100))
                            notifCenter.brightnessValue = v
                            if (typeof Settings !== "undefined") Settings.brightness = v
                        }
                    }
                }
            }
        }

        /* ═══════════════════════════════════════════════
           4. VOLUME SLIDER
           ═══════════════════════════════════════════════ */
        Rectangle {
            Layout.fillWidth: true
            height: 44
            radius: Theme.radiusTiny
            color: Theme.surfaceAlt

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 10

                Components.CanvasIcon {
                    iconName: "volume"
                    iconSize: 14
                    iconColor: Theme.textDim
                }

                /* Custom slider track + handle */
                Item {
                    Layout.fillWidth: true
                    height: 24

                    /* Track background */
                    Rectangle {
                        id: volumeTrack
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width
                        height: 4
                        radius: 2
                        color: Theme.surfaceLight

                        /* Active fill */
                        Rectangle {
                            width: parent.width * (notifCenter.volumeValue / 100)
                            height: parent.height
                            radius: 2
                            color: Theme.secondary

                            Behavior on width { NumberAnimation { duration: 60 } }
                        }
                    }

                    /* Draggable handle */
                    Rectangle {
                        id: volumeHandle
                        width: 16; height: 16; radius: 8
                        color: Theme.secondary
                        x: (notifCenter.volumeValue / 100) * (parent.width - 16)
                        anchors.verticalCenter: parent.verticalCenter

                        Behavior on x { NumberAnimation { duration: 60 } }
                    }

                    MouseArea {
                        anchors.fill: parent
                        onPressed: updateVolume(mouse)
                        onPositionChanged: if (pressed) updateVolume(mouse)
                        function updateVolume(m) {
                            var v = Math.max(0, Math.min(100, (m.x / width) * 100))
                            notifCenter.volumeValue = v
                            if (typeof Settings !== "undefined") Settings.volume = v
                        }
                    }
                }
            }
        }

        /* ═══════════════════════════════════════════════
           5. SYSTEM STATS
           ═══════════════════════════════════════════════ */
        Rectangle {
            Layout.fillWidth: true
            height: 56
            radius: Theme.radiusSmall
            color: Theme.surface

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 0

                /* CPU */
                Column {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        text: "CPU"
                        color: Theme.textDim
                        font.pixelSize: 9
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                    Text {
                        text: Math.round(SystemInfo.cpuUsage) + "%"
                        color: Theme.primary
                        font.pixelSize: 13
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                }

                /* MEM */
                Column {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        text: "MEM"
                        color: Theme.textDim
                        font.pixelSize: 9
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                    Text {
                        text: (SystemInfo.memoryUsed / 1048576).toFixed(0) + " MB"
                        color: Theme.success
                        font.pixelSize: 13
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                }

                /* TEMP */
                Column {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        text: "TEMP"
                        color: Theme.textDim
                        font.pixelSize: 9
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                    Text {
                        text: SystemInfo.cpuTemp.toFixed(0) + "\u00B0C"
                        color: SystemInfo.cpuTemp > 80 ? Theme.error
                             : SystemInfo.cpuTemp > 60 ? Theme.warning
                             : Theme.success
                        font.pixelSize: 13
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                }

                /* NPU */
                Column {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        text: "NPU"
                        color: Theme.textDim
                        font.pixelSize: 9
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                    Text {
                        text: NPUMonitor.deviceCount + " dev"
                        color: Theme.secondary
                        font.pixelSize: 13
                        font.bold: true
                        font.family: Theme.fontFamily
                    }
                }
            }
        }

        /* ═══════════════════════════════════════════════
           6. SEPARATOR
           ═══════════════════════════════════════════════ */
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: Theme.surfaceLight
        }

        /* ═══════════════════════════════════════════════
           7. NOTIFICATIONS HEADER
           ═══════════════════════════════════════════════ */
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "Notifications"
                color: Theme.text
                font.pixelSize: 13
                font.bold: true
                font.family: Theme.fontFamily
            }

            Item { Layout.fillWidth: true }

            /* Count badge */
            Rectangle {
                width: 22; height: 18; radius: 9
                color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)

                Text {
                    anchors.centerIn: parent
                    text: "5"
                    color: Theme.primary
                    font.pixelSize: 10
                    font.bold: true
                    font.family: Theme.fontFamily
                }
            }
        }

        /* ═══════════════════════════════════════════════
           8. NOTIFICATION LIST
           ═══════════════════════════════════════════════ */
        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: notifCol.height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            Column {
                id: notifCol
                width: parent.width
                spacing: 6

                Components.NotificationItem {
                    title: "System Update"
                    message: "NeuralOS v4.0 is available for download"
                    timestamp: "1m ago"
                    icon: "box"
                    iconColor: Theme.primary
                }

                Components.NotificationItem {
                    title: "NPU Active"
                    message: "Neural processing unit online \u2014 " + NPUMonitor.deviceCount + " device(s)"
                    timestamp: "3m ago"
                    icon: "chip"
                    iconColor: Theme.secondary
                }

                Components.NotificationItem {
                    title: "AI Model Loaded"
                    message: "neura_v4_core.tflite ready for inference"
                    timestamp: "8m ago"
                    icon: "neural"
                    iconColor: Theme.success
                }

                Components.NotificationItem {
                    title: "Network Connected"
                    message: "Connected to eth0 (192.168.1.100)"
                    timestamp: "22m ago"
                    icon: "wifi"
                    iconColor: Theme.primary
                }

                Components.NotificationItem {
                    title: "Security Scan"
                    message: "Full system scan complete \u2014 no threats found"
                    timestamp: "1h ago"
                    icon: "shield"
                    iconColor: Theme.success
                }
            }
        }
    }
}
