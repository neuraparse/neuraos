import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 230; height: 170
    widgetTitle: "System"

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        /* CPU */
        StatRow {
            label: "CPU"
            value: Math.round(SystemInfo.cpuUsage) + "%"
            barValue: SystemInfo.cpuUsage / 100
            barColor: SystemInfo.cpuUsage > 80 ? Theme.error :
                      SystemInfo.cpuUsage > 50 ? Theme.warning : Theme.primary
        }

        /* Memory */
        StatRow {
            label: "MEM"
            value: (SystemInfo.memoryUsed / 1048576).toFixed(0) + "M"
            barValue: SystemInfo.memoryTotal > 0 ? SystemInfo.memoryUsed / SystemInfo.memoryTotal : 0
            barColor: Theme.secondary
        }

        /* Disk */
        StatRow {
            label: "DISK"
            value: (SystemInfo.diskUsed / 1073741824).toFixed(1) + "G"
            barValue: SystemInfo.diskTotal > 0 ? SystemInfo.diskUsed / SystemInfo.diskTotal : 0
            barColor: Theme.accent
        }

        /* Temp */
        StatRow {
            label: "TEMP"
            value: SystemInfo.cpuTemp.toFixed(0) + "\u00B0C"
            barValue: Math.min(SystemInfo.cpuTemp / 85, 1)
            barColor: SystemInfo.cpuTemp > 70 ? Theme.error : Theme.success
        }
    }

    component StatRow: RowLayout {
        Layout.fillWidth: true
        spacing: 8

        property string label: ""
        property string value: ""
        property real barValue: 0
        property color barColor: Theme.primary

        Text {
            text: label; color: Theme.textDim; font.pixelSize: 10; font.bold: true
            font.family: Theme.fontFamily; Layout.preferredWidth: 34
        }

        Rectangle {
            Layout.fillWidth: true; height: 6; radius: 3
            color: Theme.surfaceLight

            Rectangle {
                width: parent.width * barValue; height: parent.height; radius: 3
                color: barColor
                Behavior on width { NumberAnimation { duration: Theme.animNormal; easing.type: Easing.OutCubic } }
            }
        }

        Text {
            text: value; color: Theme.text; font.pixelSize: 10; font.family: Theme.fontFamily
            Layout.preferredWidth: 38; horizontalAlignment: Text.AlignRight
        }
    }
}
