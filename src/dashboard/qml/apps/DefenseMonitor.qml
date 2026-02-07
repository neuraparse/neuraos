import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: defenseApp
    anchors.fill: parent

    property int critCount: 0
    property int highCount: 0
    property int medCount: 0
    property int lowCount: 0

    ListModel {
        id: threatModel
    }

    property var threatNames: [
        "Port Scan Detected", "Brute Force Attempt", "Suspicious Process",
        "Firewall Rule Triggered", "Anomaly in Traffic", "Root Login Attempt",
        "SSH Key Mismatch", "DNS Tunnel Detected", "Buffer Overflow Attempt",
        "Privilege Escalation", "Malware Signature Match", "DDoS Pattern",
        "Unauthorized Access", "Config File Modified", "ARP Spoofing Detected"
    ]

    property var categories: ["Network", "Auth", "Process", "Firewall", "System"]
    property var severities: ["Critical", "High", "Medium", "Low"]

    function getRandomSource() {
        var ifaces = NetworkManager.interfaces
        if (ifaces.length > 0) {
            var idx = Math.floor(Math.random() * ifaces.length)
            return ifaces[idx].name || "eth0"
        }
        var ips = ["192.168.1." + Math.floor(Math.random() * 254 + 1),
                   "10.0.0." + Math.floor(Math.random() * 254 + 1),
                   "203.0.113." + Math.floor(Math.random() * 254 + 1)]
        return ips[Math.floor(Math.random() * ips.length)]
    }

    function addThreat() {
        var sev = severities[Math.floor(Math.random() * severities.length)]
        threatModel.insert(0, {
            name: threatNames[Math.floor(Math.random() * threatNames.length)],
            source: getRandomSource(),
            severity: sev,
            time: Qt.formatTime(new Date(), "HH:mm:ss"),
            category: categories[Math.floor(Math.random() * categories.length)]
        })

        if (sev === "Critical") critCount++
        else if (sev === "High") highCount++
        else if (sev === "Medium") medCount++
        else lowCount++

        if (threatModel.count > 20) {
            var old = threatModel.get(threatModel.count - 1)
            if (old.severity === "Critical") critCount--
            else if (old.severity === "High") highCount--
            else if (old.severity === "Medium") medCount--
            else lowCount--
            threatModel.remove(threatModel.count - 1)
        }
    }

    Component.onCompleted: {
        for (var i = 0; i < 6; i++) addThreat()
    }

    Timer {
        interval: 5000 + Math.random() * 10000
        running: true; repeat: true
        onTriggered: {
            addThreat()
            interval = 5000 + Math.random() * 10000
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* Left: Radar + Stats */
            ColumnLayout {
                Layout.fillHeight: true
                Layout.preferredWidth: parent.width * 0.45
                spacing: 0

                /* Radar */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: Theme.darkMode ? "#0A0F1A" : "#F0F3F8"

                    Components.RadarView {
                        anchors.centerIn: parent
                        width: Math.min(parent.width, parent.height) - 20
                        height: width
                        radarColor: "#10B981"
                        targets: [
                            { x: 0.3, y: -0.2, color: "#F59E0B" },
                            { x: -0.4, y: 0.3, color: "#EF4444" },
                            { x: 0.1, y: 0.5, color: "#F59E0B" },
                            { x: -0.2, y: -0.6, color: "#10B981" },
                            { x: 0.6, y: 0.1, color: "#EF4444" }
                        ]
                    }

                    /* Overlay text */
                    Text {
                        anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 8
                        text: "THREAT RADAR | ACTIVE | " + NetworkManager.hostname
                        color: Qt.rgba(0.06, 0.72, 0.51, 0.6)
                        font.pixelSize: 9; font.family: "monospace"; font.bold: true
                    }
                }

                /* Threat stats */
                Rectangle {
                    Layout.fillWidth: true; height: 60
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.margins: 10; spacing: 14

                        ThreatStat { label: "Critical"; count: critCount.toString(); statColor: "#EF4444" }
                        ThreatStat { label: "High"; count: highCount.toString(); statColor: "#F59E0B" }
                        ThreatStat { label: "Medium"; count: medCount.toString(); statColor: "#3B82F6" }
                        ThreatStat { label: "Low"; count: lowCount.toString(); statColor: "#6B7280" }
                        Item { Layout.fillWidth: true }
                        ThreatStat { label: "Total"; count: threatModel.count.toString(); statColor: Theme.primary }
                    }
                }
            }

            Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

            /* Right: Threat list + timeline */
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* Header */
                Rectangle {
                    Layout.fillWidth: true; height: 40
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.margins: 10

                        Text { text: "Threat Log"; color: Theme.text; font.pixelSize: 13; font.bold: true }
                        Item { Layout.fillWidth: true }

                        Rectangle {
                            width: 8; height: 8; radius: 4
                            color: "#EF4444"
                            SequentialAnimation on opacity {
                                loops: Animation.Infinite
                                NumberAnimation { to: 0.3; duration: 800 }
                                NumberAnimation { to: 1; duration: 800 }
                            }
                        }
                        Text { text: "MONITORING"; color: Theme.error; font.pixelSize: 9; font.bold: true; font.family: "monospace" }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                /* Threat list */
                Flickable {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentHeight: threatCol.height
                    clip: true

                    Column {
                        id: threatCol
                        width: parent.width
                        spacing: 1

                        Repeater {
                            model: threatModel

                            Rectangle {
                                width: parent.width; height: 56
                                color: threatItemMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 8

                                    /* Severity indicator */
                                    Rectangle {
                                        width: 4; height: 36; radius: 2
                                        color: model.severity === "Critical" ? "#EF4444" :
                                               model.severity === "High" ? "#F59E0B" :
                                               model.severity === "Medium" ? "#3B82F6" : "#6B7280"
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: model.name; color: Theme.text; font.pixelSize: 11; font.bold: true }
                                        Text { text: "Source: " + model.source + " | " + model.category; color: Theme.textDim; font.pixelSize: 9 }
                                    }

                                    Column {
                                        spacing: 2
                                        Components.StatusBadge {
                                            text: model.severity
                                            badgeColor: model.severity === "Critical" ? "#EF4444" :
                                                        model.severity === "High" ? "#F59E0B" :
                                                        model.severity === "Medium" ? "#3B82F6" : "#6B7280"
                                        }
                                        Text { text: model.time; color: Theme.textMuted; font.pixelSize: 8; font.family: "monospace" }
                                    }
                                }

                                MouseArea { id: threatItemMa; anchors.fill: parent; hoverEnabled: true }
                            }
                        }
                    }
                }

                /* Actions bar */
                Rectangle {
                    Layout.fillWidth: true; height: 44
                    color: Theme.surface

                    Row {
                        anchors.centerIn: parent; spacing: 8

                        Repeater {
                            model: ["Block All", "Scan Now", "Export Log", "Clear Alerts"]

                            Rectangle {
                                width: 85; height: 30; radius: Theme.radiusSmall
                                color: modelData === "Block All" ?
                                    Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.2) :
                                    daMa.containsMouse ? Theme.surfaceAlt : Theme.surface

                                Text {
                                    anchors.centerIn: parent; text: modelData
                                    color: modelData === "Block All" ? Theme.error : Theme.text
                                    font.pixelSize: 10
                                }
                                MouseArea {
                                    id: daMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (modelData === "Clear Alerts") {
                                            threatModel.clear()
                                            critCount = 0; highCount = 0; medCount = 0; lowCount = 0
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

    component ThreatStat: Column {
        spacing: 2
        property string label: ""
        property string count: ""
        property color statColor: Theme.text

        Text { text: count; color: statColor; font.pixelSize: 16; font.bold: true; anchors.horizontalCenter: parent.horizontalCenter }
        Text { text: label; color: Theme.textDim; font.pixelSize: 8; anchors.horizontalCenter: parent.horizontalCenter }
    }
}
