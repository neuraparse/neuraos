import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 210; height: 140
    widgetTitle: "AI Engine"

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Rectangle {
                width: 8; height: 8; radius: 4
                color: NPIE.modelLoaded ? Theme.success : Theme.textDim

                SequentialAnimation on opacity {
                    running: NPIE.modelLoaded
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.4; duration: 800 }
                    NumberAnimation { to: 1.0; duration: 800 }
                }
            }

            Text {
                text: NPIE.modelLoaded ? "Model Active" : "Idle"
                color: Theme.text; font.pixelSize: 12; font.bold: true
                font.family: Theme.fontFamily
            }
        }

        /* Stats rows */
        InfoRow { label: "Backend:"; value: NPIE.currentBackend; clr: Theme.primary }
        InfoRow { label: "Inferences:"; value: NPIE.inferenceCount.toString(); clr: Theme.success }
        InfoRow { label: "Latency:"; value: NPIE.avgInferenceMs.toFixed(1) + " ms"; clr: Theme.warning }
        InfoRow { label: "NPU:"; value: NPUMonitor.deviceCount > 0 ? NPUMonitor.deviceCount + " active" : "None"; clr: Theme.secondary }
    }

    component InfoRow: RowLayout {
        Layout.fillWidth: true
        spacing: 6
        property string label: ""
        property string value: ""
        property color clr: Theme.text

        Text { text: label; color: Theme.textDim; font.pixelSize: 10; font.family: Theme.fontFamily }
        Text { text: value; color: clr; font.pixelSize: 10; font.bold: true; font.family: Theme.fontFamily }
    }
}
