import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: droneApp
    anchors.fill: parent

    property int selectedDrone: 0

    ListModel {
        id: droneFleet
        ListElement { name: "Alpha-01"; status: "Airborne"; battery: 87; alt: 120; spd: 15; lat: 34.052; lon: -118.243 }
        ListElement { name: "Bravo-02"; status: "Airborne"; battery: 72; alt: 85; spd: 22; lat: 34.055; lon: -118.240 }
        ListElement { name: "Charlie-03"; status: "Grounded"; battery: 95; alt: 0; spd: 0; lat: 34.050; lon: -118.245 }
        ListElement { name: "Delta-04"; status: "Landing"; battery: 31; alt: 25; spd: 5; lat: 34.048; lon: -118.241 }
        ListElement { name: "Echo-05"; status: "Airborne"; battery: 64; alt: 200; spd: 30; lat: 34.060; lon: -118.238 }
    }

    /* Battery drain + dynamic simulation */
    Timer {
        interval: 3000; running: true; repeat: true
        onTriggered: {
            for (var i = 0; i < droneFleet.count; i++) {
                var d = droneFleet.get(i)
                if (d.status === "Airborne") {
                    var newBat = Math.max(0, d.battery - Math.floor(Math.random() * 2 + 1))
                    var newAlt = d.alt + Math.floor(Math.random() * 11 - 5)
                    var newSpd = Math.max(0, d.spd + Math.floor(Math.random() * 7 - 3))
                    var newLat = d.lat + (Math.random() * 0.002 - 0.001)
                    var newLon = d.lon + (Math.random() * 0.002 - 0.001)
                    droneFleet.setProperty(i, "battery", newBat)
                    droneFleet.setProperty(i, "alt", Math.max(10, newAlt))
                    droneFleet.setProperty(i, "spd", newSpd)
                    droneFleet.setProperty(i, "lat", newLat)
                    droneFleet.setProperty(i, "lon", newLon)
                    if (newBat <= 10) {
                        droneFleet.setProperty(i, "status", "Landing")
                        droneFleet.setProperty(i, "spd", 3)
                    }
                } else if (d.status === "Landing") {
                    var landAlt = Math.max(0, d.alt - 5)
                    droneFleet.setProperty(i, "alt", landAlt)
                    droneFleet.setProperty(i, "battery", Math.max(0, d.battery - 1))
                    if (landAlt <= 0) {
                        droneFleet.setProperty(i, "status", "Grounded")
                        droneFleet.setProperty(i, "alt", 0)
                        droneFleet.setProperty(i, "spd", 0)
                    }
                } else if (d.status === "Grounded" && d.battery < 95) {
                    droneFleet.setProperty(i, "battery", Math.min(100, d.battery + 3))
                }
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* Left panel: Map + Fleet */
            ColumnLayout {
                Layout.fillHeight: true
                Layout.preferredWidth: parent.width * 0.6
                spacing: 0

                /* Map Canvas */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: Theme.darkMode ? "#0A1628" : "#EEF1F7"

                    Canvas {
                        id: mapCanvas
                        anchors.fill: parent

                        property int tick: 0

                        Timer {
                            interval: 1000; running: true; repeat: true
                            onTriggered: { mapCanvas.tick++; mapCanvas.requestPaint() }
                        }

                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)

                            /* Grid */
                            ctx.strokeStyle = Qt.rgba(0, 0.85, 1, 0.06)
                            ctx.lineWidth = 1
                            var gridSize = 40
                            for (var gx = 0; gx < width; gx += gridSize) {
                                ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, height); ctx.stroke()
                            }
                            for (var gy = 0; gy < height; gy += gridSize) {
                                ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke()
                            }

                            /* Drone positions (simulated) */
                            for (var i = 0; i < droneFleet.count; i++) {
                                var d = droneFleet.get(i)
                                var dx = width * 0.2 + (i * width * 0.15) + Math.sin(tick * 0.1 + i) * 10
                                var dy = height * 0.3 + (i % 3) * height * 0.15 + Math.cos(tick * 0.08 + i) * 8

                                /* Trail */
                                if (d.status === "Airborne") {
                                    ctx.beginPath()
                                    ctx.arc(dx, dy, 18, 0, Math.PI * 2)
                                    ctx.fillStyle = Qt.rgba(0, 0.85, 1, 0.05)
                                    ctx.fill()
                                }

                                /* Drone dot */
                                var clr = d.status === "Airborne" ? "#10B981" :
                                          d.status === "Landing" ? "#F59E0B" : "#6B7280"
                                ctx.beginPath()
                                ctx.arc(dx, dy, 5, 0, Math.PI * 2)
                                ctx.fillStyle = clr
                                ctx.fill()

                                /* Label */
                                ctx.fillStyle = Theme.darkMode ? "#FFFFFF" : "#1A1A2E"
                                ctx.font = "9px monospace"
                                ctx.fillText(d.name, dx + 8, dy + 3)

                                /* Selected ring */
                                if (i === selectedDrone) {
                                    ctx.beginPath()
                                    ctx.arc(dx, dy, 12, 0, Math.PI * 2)
                                    ctx.strokeStyle = "#00D9FF"
                                    ctx.lineWidth = 2
                                    ctx.stroke()
                                }
                            }

                            /* Waypoint lines between active drones */
                            ctx.strokeStyle = Qt.rgba(0, 0.85, 1, 0.15)
                            ctx.lineWidth = 1
                            ctx.setLineDash([4, 4])
                            var prevX = -1, prevY = -1
                            for (var j = 0; j < droneFleet.count; j++) {
                                if (droneFleet.get(j).status === "Airborne") {
                                    var jx = width * 0.2 + (j * width * 0.15) + Math.sin(tick * 0.1 + j) * 10
                                    var jy = height * 0.3 + (j % 3) * height * 0.15 + Math.cos(tick * 0.08 + j) * 8
                                    if (prevX >= 0) {
                                        ctx.beginPath(); ctx.moveTo(prevX, prevY); ctx.lineTo(jx, jy); ctx.stroke()
                                    }
                                    prevX = jx; prevY = jy
                                }
                            }
                            ctx.setLineDash([])
                        }
                    }

                    /* Map overlay info */
                    Text {
                        anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 8
                        text: "TACTICAL MAP | " + droneFleet.count + " units"
                        color: Qt.rgba(0, 0.85, 1, 0.5); font.pixelSize: 9; font.family: "monospace"
                    }
                }

                /* Fleet list */
                Rectangle {
                    Layout.fillWidth: true; height: 200
                    color: Theme.surface

                    Column {
                        anchors.fill: parent; anchors.margins: 6; spacing: 2

                        Text { text: "Fleet Status"; color: Theme.textDim; font.pixelSize: 10; font.bold: true; bottomPadding: 4 }

                        Repeater {
                            model: droneFleet

                            Components.DroneIndicator {
                                droneName: model.name
                                droneStatus: model.status
                                batteryPercent: model.battery
                                altitude: model.alt
                                speed: model.spd
                                onSelected: selectedDrone = index
                            }
                        }
                    }
                }
            }

            Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

            /* Right panel: Selected drone telemetry */
            Rectangle {
                Layout.fillWidth: true; Layout.fillHeight: true
                color: Theme.background

                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                    Text {
                        text: droneFleet.get(selectedDrone).name + " Telemetry"
                        color: Theme.text; font.pixelSize: 14; font.bold: true
                    }

                    Components.StatusBadge {
                        text: droneFleet.get(selectedDrone).status
                        badgeColor: droneFleet.get(selectedDrone).status === "Airborne" ? Theme.success :
                                    droneFleet.get(selectedDrone).status === "Landing" ? Theme.warning : Theme.textDim
                    }

                    /* Telemetry grid */
                    GridLayout {
                        Layout.fillWidth: true; columns: 3; rowSpacing: 6; columnSpacing: 6

                        TelemetryCard { label: "Altitude"; value: droneFleet.get(selectedDrone).alt + " m"; cardColor: Theme.primary }
                        TelemetryCard { label: "Speed"; value: droneFleet.get(selectedDrone).spd + " m/s"; cardColor: Theme.success }
                        TelemetryCard { label: "Battery"; value: droneFleet.get(selectedDrone).battery + "%"; cardColor: droneFleet.get(selectedDrone).battery > 50 ? Theme.success : Theme.warning }
                        TelemetryCard { label: "GPS"; value: droneFleet.get(selectedDrone).lat.toFixed(3) + ", " + droneFleet.get(selectedDrone).lon.toFixed(3); cardColor: Theme.secondary }
                        TelemetryCard { label: "Op Temp"; value: SystemInfo.cpuTemp.toFixed(0) + "\u00B0C"; cardColor: SystemInfo.cpuTemp > 70 ? Theme.error : Theme.warning }
                        TelemetryCard { label: "CPU Load"; value: Math.round(SystemInfo.cpuUsage) + "%"; cardColor: SystemInfo.cpuUsage > 80 ? Theme.error : Theme.primary }
                    }

                    /* Controls */
                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                    Text { text: "Commands"; color: Theme.textDim; font.pixelSize: 10; font.bold: true }

                    GridLayout {
                        Layout.fillWidth: true; columns: 2; rowSpacing: 6; columnSpacing: 6

                        Repeater {
                            model: ["Launch", "Land", "Return Home", "Hold Position", "Set Waypoint", "Emergency Stop"]

                            Rectangle {
                                Layout.fillWidth: true; height: 32; radius: Theme.radiusSmall
                                color: modelData === "Emergency Stop" ?
                                    Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.2) :
                                    cmdMa.containsMouse ? Theme.surfaceAlt : Theme.surface

                                Text {
                                    anchors.centerIn: parent; text: modelData
                                    color: modelData === "Emergency Stop" ? Theme.error : Theme.text
                                    font.pixelSize: 10
                                }
                                MouseArea { id: cmdMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }

    component TelemetryCard: Rectangle {
        Layout.fillWidth: true; height: 50
        radius: Theme.radiusSmall; color: Theme.surface

        property string label: ""
        property string value: ""
        property color cardColor: Theme.primary

        Column {
            anchors.centerIn: parent; spacing: 2
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: label; color: Theme.textDim; font.pixelSize: 9; font.bold: true }
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: value; color: cardColor; font.pixelSize: 14; font.bold: true }
        }
    }
}
