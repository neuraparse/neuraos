import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."

Rectangle {
    id: droneInd
    width: parent ? parent.width : 200
    height: 50
    radius: Theme.radiusSmall
    color: indMa.containsMouse ? Theme.surfaceAlt : Theme.surface

    property string droneName: "Drone-01"
    property string droneStatus: "Airborne"
    property int batteryPercent: 85
    property real altitude: 120.0
    property real speed: 15.5

    signal selected()

    RowLayout {
        anchors.fill: parent; anchors.margins: 8; spacing: 8

        Rectangle {
            width: 32; height: 32; radius: 16
            color: droneStatus === "Airborne" ? Qt.rgba(0.06, 0.72, 0.51, 0.15) :
                   droneStatus === "Landing" ? Qt.rgba(0.96, 0.62, 0.04, 0.15) :
                   Qt.rgba(0.5, 0.5, 0.5, 0.1)

            CanvasIcon {
                anchors.centerIn: parent; iconName: "drone"; iconSize: 16
                iconColor: droneStatus === "Airborne" ? Theme.success :
                           droneStatus === "Landing" ? Theme.warning : Theme.textDim
            }
        }

        Column {
            Layout.fillWidth: true; spacing: 1
            Text { text: droneName; color: Theme.text; font.pixelSize: 11; font.bold: true }
            Text { text: droneStatus + " | " + altitude.toFixed(0) + "m | " + speed.toFixed(0) + " m/s"; color: Theme.textDim; font.pixelSize: 9 }
        }

        /* Battery */
        Row {
            spacing: 4
            Rectangle {
                width: 24; height: 10; radius: 2; anchors.verticalCenter: parent.verticalCenter
                color: "transparent"; border.width: 1; border.color: Theme.textDim

                Rectangle {
                    x: 1; y: 1; width: (parent.width - 2) * batteryPercent / 100; height: parent.height - 2
                    color: batteryPercent > 50 ? Theme.success : batteryPercent > 20 ? Theme.warning : Theme.error
                }
            }
            Text { text: batteryPercent + "%"; color: Theme.textDim; font.pixelSize: 9; anchors.verticalCenter: parent.verticalCenter }
        }
    }

    MouseArea {
        id: indMa; anchors.fill: parent
        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
        onClicked: droneInd.selected()
    }
}
