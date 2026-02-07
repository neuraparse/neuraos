import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: homePage

    Flickable {
        anchors.fill: parent
        anchors.margins: 20
        contentHeight: mainCol.height
        clip: true

        ColumnLayout {
            id: mainCol
            width: parent.width
            spacing: 20

            /* Clock & Date */
            Column {
                Layout.alignment: Qt.AlignHCenter
                spacing: 4

                Text {
                    id: bigClock
                    anchors.horizontalCenter: parent.horizontalCenter
                    color: Theme.text
                    font.pixelSize: Theme.fontSizeHuge
                    font.bold: true

                    Timer {
                        interval: 1000; running: true; repeat: true; triggeredOnStart: true
                        onTriggered: bigClock.text = Qt.formatTime(new Date(), "HH:mm:ss")
                    }
                }

                Text {
                    id: dateText
                    anchors.horizontalCenter: parent.horizontalCenter
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeLarge

                    Timer {
                        interval: 60000; running: true; repeat: true; triggeredOnStart: true
                        onTriggered: dateText.text = Qt.formatDate(new Date(), "dddd, MMMM d, yyyy")
                    }
                }
            }

            /* System Gauges */
            Components.SectionHeader { title: "System Status" }

            Row {
                Layout.alignment: Qt.AlignHCenter
                spacing: 30

                Components.CircularGauge {
                    width: 110; height: 110
                    value: SystemInfo.cpuUsage
                    gaugeColor: Theme.primary
                    label: "CPU"
                }

                Components.CircularGauge {
                    width: 110; height: 110
                    value: SystemInfo.memoryTotal > 0 ?
                        SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
                    gaugeColor: Theme.success
                    label: "Memory"
                }

                Components.CircularGauge {
                    width: 110; height: 110
                    value: SystemInfo.diskTotal > 0 ?
                        SystemInfo.diskUsed / SystemInfo.diskTotal * 100 : 0
                    gaugeColor: Theme.warning
                    label: "Disk"
                }

                Components.CircularGauge {
                    width: 110; height: 110
                    value: SystemInfo.cpuTemp
                    gaugeColor: SystemInfo.cpuTemp > 70 ? Theme.error : Theme.secondary
                    label: "Temp"
                    unit: "\u00B0C"
                }
            }

            /* Info Cards Row */
            Components.SectionHeader { title: "Quick Info" }

            GridLayout {
                Layout.fillWidth: true
                columns: 4
                columnSpacing: 12
                rowSpacing: 12

                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Hostname"
                    value: SystemInfo.hostname
                    unit: ""
                    icon: "\u2302"
                    accentColor: Theme.primary
                }

                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Kernel"
                    value: SystemInfo.kernelVersion
                    unit: ""
                    icon: "\u2699"
                    accentColor: Theme.secondary
                }

                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "Uptime"
                    value: SystemInfo.uptime
                    unit: ""
                    icon: "\u23F1"
                    accentColor: Theme.success
                }

                Components.MetricCard {
                    Layout.fillWidth: true
                    title: "NPU Devices"
                    value: NPUMonitor.deviceCount.toString()
                    unit: "detected"
                    icon: "\u2756"
                    accentColor: Theme.warning
                }
            }

            /* Quick Actions */
            Components.SectionHeader { title: "Quick Actions" }

            Row {
                Layout.alignment: Qt.AlignHCenter
                spacing: 20

                Repeater {
                    model: ListModel {
                        ListElement { label: "AI Dashboard"; icon: "\u2699"; page: 1; clr: "#00D9FF" }
                        ListElement { label: "Terminal";     icon: "\u2756"; page: 3; clr: "#10B981" }
                        ListElement { label: "Settings";     icon: "\u2731"; page: 5; clr: "#7C3AED" }
                        ListElement { label: "All Apps";     icon: "\u2637"; page: 6; clr: "#F59E0B" }
                    }

                    delegate: Rectangle {
                        width: 130; height: 80
                        radius: Theme.radius
                        color: Theme.surface
                        border.width: 1
                        border.color: ma.containsMouse ? model.clr : Theme.surfaceLight

                        Column {
                            anchors.centerIn: parent
                            spacing: 6

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: model.icon
                                font.pixelSize: 24
                                color: model.clr
                            }
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: model.label
                                font.pixelSize: Theme.fontSizeSmall
                                color: Theme.text
                            }
                        }

                        MouseArea {
                            id: ma
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                homePage.parent.parent.currentIndex = model.page
                            }
                        }

                        Behavior on border.color { ColorAnimation { duration: Theme.animFast } }
                    }
                }
            }

            /* NPIE Info */
            Components.SectionHeader { title: "AI Runtime" }

            Rectangle {
                Layout.fillWidth: true
                height: 60
                radius: Theme.radius
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 20

                    Text { text: "NPIE Version:"; color: Theme.textDim; font.pixelSize: Theme.fontSizeNormal }
                    Text { text: NPIE.version; color: Theme.primary; font.pixelSize: Theme.fontSizeNormal; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Text { text: "Backend:"; color: Theme.textDim; font.pixelSize: Theme.fontSizeNormal }
                    Text { text: NPIE.currentBackend; color: Theme.success; font.pixelSize: Theme.fontSizeNormal; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Components.StatusBadge { text: "Online"; badgeColor: Theme.success }
                }
            }

            Item { height: 20 } /* Bottom spacing */
        }
    }
}
