import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: notifCenter
    width: Theme.notifCenterW
    color: Theme.startMenuBg
    radius: Theme.radiusSmall
    border.width: 1
    border.color: Theme.surfaceLight

    signal closePanel()

    /* Quick toggles + notifications */
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        /* Header */
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "Notifications"
                color: Theme.text
                font.pixelSize: 14
                font.bold: true
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                width: 28; height: 28; radius: 14
                color: clearMa.containsMouse ? Theme.surfaceAlt : "transparent"
                Components.CanvasIcon { anchors.centerIn: parent; iconName: "close"; iconColor: Theme.textDim; iconSize: 14 }
                MouseArea {
                    id: clearMa; anchors.fill: parent
                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: notifCenter.closePanel()
                }
            }
        }

        /* Quick Toggles */
        Rectangle {
            Layout.fillWidth: true
            height: 80
            radius: Theme.radius
            color: Theme.surface

            Text {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.margins: 10
                text: "Quick Settings"
                color: Theme.textDim
                font.pixelSize: 10
                font.bold: true
            }

            Row {
                anchors.centerIn: parent
                anchors.verticalCenterOffset: 6
                spacing: 10

                /* WiFi */
                Rectangle {
                    width: 50; height: 44; radius: Theme.radiusSmall
                    color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                    Column { anchors.centerIn: parent; spacing: 2
                        Components.CanvasIcon { anchors.horizontalCenter: parent.horizontalCenter; iconName: "wifi"; iconSize: 14; iconColor: Theme.primary }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "WiFi"; font.pixelSize: 8; color: Theme.textDim }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                }
                /* BT */
                Rectangle {
                    width: 50; height: 44; radius: Theme.radiusSmall; color: Theme.surfaceAlt
                    Column { anchors.centerIn: parent; spacing: 2
                        Components.CanvasIcon { anchors.horizontalCenter: parent.horizontalCenter; iconName: "bluetooth"; iconSize: 14; iconColor: Theme.textDim }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "BT"; font.pixelSize: 8; color: Theme.textDim }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                }
                /* NPU */
                Rectangle {
                    width: 50; height: 44; radius: Theme.radiusSmall
                    color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                    Column { anchors.centerIn: parent; spacing: 2
                        Components.CanvasIcon { anchors.horizontalCenter: parent.horizontalCenter; iconName: "chip"; iconSize: 14; iconColor: Theme.primary }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "NPU"; font.pixelSize: 8; color: Theme.textDim }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                }
                /* DND */
                Rectangle {
                    width: 50; height: 44; radius: Theme.radiusSmall; color: Theme.surfaceAlt
                    Column { anchors.centerIn: parent; spacing: 2
                        Components.CanvasIcon { anchors.horizontalCenter: parent.horizontalCenter; iconName: "moon"; iconSize: 14; iconColor: Theme.textDim }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "DND"; font.pixelSize: 8; color: Theme.textDim }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                }
                /* Theme Toggle */
                Rectangle {
                    width: 50; height: 44; radius: Theme.radiusSmall
                    color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                    Column { anchors.centerIn: parent; spacing: 2
                        Components.CanvasIcon { anchors.horizontalCenter: parent.horizontalCenter; iconName: Theme.darkMode ? "moon" : "sun"; iconSize: 14; iconColor: Theme.primary }
                        Text { anchors.horizontalCenter: parent.horizontalCenter; text: Theme.darkMode ? "Dark" : "Light"; font.pixelSize: 8; color: Theme.textDim }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: Settings.theme = (Settings.theme === "dark" ? "light" : "dark") }
                }
            }
        }

        /* System stats */
        Rectangle {
            Layout.fillWidth: true
            height: 50
            radius: Theme.radius
            color: Theme.surface

            RowLayout {
                anchors.fill: parent; anchors.margins: 10; spacing: 14

                Column {
                    spacing: 2
                    Text { text: "CPU"; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
                    Text { text: Math.round(SystemInfo.cpuUsage) + "%"; color: Theme.primary; font.pixelSize: 12; font.bold: true }
                }

                Column {
                    spacing: 2
                    Text { text: "MEM"; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
                    Text {
                        text: (SystemInfo.memoryUsed / 1048576).toFixed(0) + "M"
                        color: Theme.success; font.pixelSize: 12; font.bold: true
                    }
                }

                Column {
                    spacing: 2
                    Text { text: "TEMP"; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
                    Text {
                        text: SystemInfo.cpuTemp.toFixed(0) + "\u00B0"
                        color: SystemInfo.cpuTemp > 70 ? Theme.error : Theme.warning
                        font.pixelSize: 12; font.bold: true
                    }
                }

                Item { Layout.fillWidth: true }

                Column {
                    spacing: 2
                    Text { text: "NPU"; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
                    Text { text: NPUMonitor.deviceCount + " dev"; color: Theme.secondary; font.pixelSize: 12; font.bold: true }
                }
            }
        }

        /* Separator */
        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

        /* Notification list */
        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: notifCol.height
            clip: true

            Column {
                id: notifCol
                width: parent.width
                spacing: 6

                Components.NotificationItem {
                    title: "System Update"
                    message: "NeuralOS 3.2.0 is running"
                    timestamp: "2m ago"
                    icon: "box"
                    iconColor: Theme.primary
                }

                Components.NotificationItem {
                    title: "NPU Active"
                    message: "Neural processing unit initialized"
                    timestamp: "5m ago"
                    icon: "chip"
                    iconColor: Theme.secondary
                }

                Components.NotificationItem {
                    title: "Model Loaded"
                    message: "demo_model.tflite ready for inference"
                    timestamp: "12m ago"
                    icon: "neural"
                    iconColor: Theme.success
                }

                Components.NotificationItem {
                    title: "Network Connected"
                    message: "Connected to eth0 (192.168.1.100)"
                    timestamp: "1h ago"
                    icon: "wifi"
                    iconColor: Theme.primary
                }

                Components.NotificationItem {
                    title: "Security Scan Complete"
                    message: "No threats detected"
                    timestamp: "2h ago"
                    icon: "shield"
                    iconColor: Theme.success
                }
            }
        }
    }
}
