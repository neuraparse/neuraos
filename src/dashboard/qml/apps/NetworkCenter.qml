import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: netApp
    anchors.fill: parent

    property int activeTab: 0

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Tabs */
            Rectangle {
                Layout.fillWidth: true; height: 40
                color: Theme.surface

                Row {
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: parent.left; anchors.leftMargin: 10
                    spacing: 2

                    Repeater {
                        model: ["Interfaces", "Firewall", "DNS", "VPN"]

                        Rectangle {
                            width: 80; height: 28; radius: Theme.radiusSmall
                            color: activeTab === index ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2) :
                                   ntMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent; text: modelData
                                color: activeTab === index ? Theme.primary : Theme.text
                                font.pixelSize: 11; font.bold: activeTab === index
                            }
                            MouseArea { id: ntMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: activeTab = index }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            Flickable {
                Layout.fillWidth: true; Layout.fillHeight: true
                contentHeight: netContent.height; clip: true

                Column {
                    id: netContent
                    width: parent.width

                    /* ─── Interfaces ─── */
                    ColumnLayout {
                        width: parent.width; spacing: 8; visible: activeTab === 0

                        Item { height: 10 }

                        /* Hostname */
                        Rectangle {
                            Layout.fillWidth: true; Layout.margins: 10; height: 48
                            radius: Theme.radius; color: Theme.surface

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 12

                                Text { text: "Hostname"; color: Theme.textDim; font.pixelSize: 11 }
                                Item { Layout.fillWidth: true }
                                Text { text: NetworkManager.hostname; color: Theme.primary; font.pixelSize: 12; font.bold: true }
                            }
                        }

                        /* Interface list */
                        Repeater {
                            model: NetworkManager.interfaces

                            Rectangle {
                                Layout.fillWidth: true; Layout.margins: 10; height: 70
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 12

                                    Rectangle {
                                        width: 40; height: 40; radius: 20
                                        color: modelData.isUp ? Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15) :
                                               Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)
                                        Text {
                                            anchors.centerIn: parent; text: "\u2301"; font.pixelSize: 18
                                            color: modelData.isUp ? Theme.success : Theme.error
                                        }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: modelData.name || "Unknown"; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                        Text { text: "Type: " + (modelData.type || "N/A") + " | Addr: " + (modelData.address || "N/A"); color: Theme.textDim; font.pixelSize: 10 }
                                    }

                                    Components.StatusBadge {
                                        text: modelData.isUp ? "Up" : "Down"
                                        badgeColor: modelData.isUp ? Theme.success : Theme.error
                                    }
                                }
                            }
                        }

                        Item { height: 8 }
                    }

                    /* ─── Firewall ─── */
                    ColumnLayout {
                        width: parent.width; spacing: 8; visible: activeTab === 1

                        Item { height: 10 }

                        Repeater {
                            model: ListModel {
                                ListElement { rule: "Allow SSH (22)"; direction: "Inbound"; proto: "TCP"; status: "Active" }
                                ListElement { rule: "Allow HTTP (80)"; direction: "Inbound"; proto: "TCP"; status: "Active" }
                                ListElement { rule: "Allow HTTPS (443)"; direction: "Inbound"; proto: "TCP"; status: "Active" }
                                ListElement { rule: "Allow NPIE API (8080)"; direction: "Inbound"; proto: "TCP"; status: "Active" }
                                ListElement { rule: "Block Telnet (23)"; direction: "Inbound"; proto: "TCP"; status: "Active" }
                                ListElement { rule: "Allow DNS (53)"; direction: "Outbound"; proto: "UDP"; status: "Active" }
                                ListElement { rule: "Allow NTP (123)"; direction: "Outbound"; proto: "UDP"; status: "Active" }
                            }

                            Rectangle {
                                Layout.fillWidth: true; Layout.margins: 10; height: 48
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                                    Rectangle {
                                        width: 4; height: 28; radius: 2
                                        color: model.rule.indexOf("Block") >= 0 ? Theme.error : Theme.success
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: model.rule; color: Theme.text; font.pixelSize: 11; font.bold: true }
                                        Text { text: model.direction + " | " + model.proto; color: Theme.textDim; font.pixelSize: 9 }
                                    }

                                    Components.StatusBadge { text: model.status; badgeColor: Theme.success }
                                }
                            }
                        }

                        Item { height: 8 }
                    }

                    /* ─── DNS ─── */
                    ColumnLayout {
                        width: parent.width; spacing: 8; visible: activeTab === 2

                        Item { height: 10 }

                        Repeater {
                            model: ListModel {
                                ListElement { server: "1.1.1.1"; provider: "Cloudflare"; latency: "4ms"; active: true }
                                ListElement { server: "8.8.8.8"; provider: "Google"; latency: "12ms"; active: false }
                                ListElement { server: "9.9.9.9"; provider: "Quad9"; latency: "8ms"; active: false }
                            }

                            Rectangle {
                                Layout.fillWidth: true; Layout.margins: 10; height: 50
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                                    Rectangle {
                                        width: 8; height: 8; radius: 4
                                        color: model.active ? Theme.success : Theme.textDim
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: model.server; color: Theme.text; font.pixelSize: 12; font.bold: true; font.family: "monospace" }
                                        Text { text: model.provider + " | " + model.latency; color: Theme.textDim; font.pixelSize: 10 }
                                    }

                                    Components.StatusBadge {
                                        text: model.active ? "Primary" : "Standby"
                                        badgeColor: model.active ? Theme.primary : Theme.textDim
                                    }
                                }
                            }
                        }

                        Item { height: 8 }
                    }

                    /* ─── VPN ─── */
                    ColumnLayout {
                        width: parent.width; spacing: 8; visible: activeTab === 3

                        Item { height: 10 }

                        Rectangle {
                            Layout.fillWidth: true; Layout.margins: 10; height: 100
                            radius: Theme.radius; color: Theme.surface

                            ColumnLayout {
                                anchors.fill: parent; anchors.margins: 14; spacing: 8

                                RowLayout {
                                    Text { text: "VPN Status"; color: Theme.text; font.pixelSize: 12; font.bold: true }
                                    Item { Layout.fillWidth: true }
                                    Components.StatusBadge { text: "Disconnected"; badgeColor: Theme.textDim }
                                }

                                Text { text: "No active VPN connection. Configure a tunnel to secure network traffic."; color: Theme.textDim; font.pixelSize: 10; wrapMode: Text.Wrap }

                                Rectangle {
                                    width: 120; height: 30; radius: Theme.radiusSmall
                                    color: vpnMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary
                                    Text { anchors.centerIn: parent; text: "Connect VPN"; color: "#000"; font.bold: true; font.pixelSize: 10 }
                                    MouseArea { id: vpnMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                                }
                            }
                        }

                        Repeater {
                            model: ListModel {
                                ListElement { name: "WireGuard Tunnel"; endpoint: "vpn.example.com:51820"; proto: "WireGuard" }
                                ListElement { name: "Office VPN"; endpoint: "office.corp:1194"; proto: "OpenVPN" }
                            }

                            Rectangle {
                                Layout.fillWidth: true; Layout.margins: 10; height: 50
                                radius: Theme.radius; color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent; anchors.margins: 12; spacing: 10

                                    Column {
                                        Layout.fillWidth: true; spacing: 2
                                        Text { text: model.name; color: Theme.text; font.pixelSize: 11; font.bold: true }
                                        Text { text: model.endpoint + " | " + model.proto; color: Theme.textDim; font.pixelSize: 9 }
                                    }

                                    Rectangle {
                                        width: 60; height: 24; radius: Theme.radiusSmall
                                        color: Theme.surfaceAlt
                                        Text { anchors.centerIn: parent; text: "Connect"; color: Theme.primary; font.pixelSize: 9 }
                                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                                    }
                                }
                            }
                        }

                        Item { height: 8 }
                    }
                }
            }
        }
    }
}
