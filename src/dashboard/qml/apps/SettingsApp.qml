import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: settingsApp
    anchors.fill: parent

    property string selectedCategory: "system"

    /* ─── Category model ─── */
    ListModel {
        id: categoryModel
        ListElement { key: "system";  label: "System";   ico: "gear" }
        ListElement { key: "display"; label: "Display";  ico: "monitor" }
        ListElement { key: "sound";   label: "Sound";    ico: "volume" }
        ListElement { key: "network"; label: "Network";  ico: "wifi" }
        ListElement { key: "ainpu";   label: "AI & NPU"; ico: "chip" }
        ListElement { key: "privacy"; label: "Privacy";  ico: "shield" }
        ListElement { key: "apps";    label: "Apps";     ico: "grid" }
        ListElement { key: "storage"; label: "Storage";  ico: "folder" }
        ListElement { key: "users";   label: "Users";    ico: "user" }
        ListElement { key: "about";   label: "About";    ico: "info" }
    }

    /* ─── Local state properties ─── */
    property bool performanceMode: false
    property real animationSpeed: 1.0
    property string selectedWallpaperColor: "#0D0D12"
    property real displayScale: 100
    property real masterVolume: 75
    property bool notificationSounds: true
    property bool systemSounds: true
    property string connectedNetwork: "NeuralOS-5G"
    property string ipAddress: "192.168.1.42"
    property bool proxyEnabled: false
    property real modelCacheSize: 2.4
    property bool autoInference: true
    property string npuPowerMode: "balanced"
    property bool firewallEnabled: true
    property bool telemetryEnabled: false
    property bool screenLockEnabled: true
    property real cacheSize: 1.8
    property string currentHostname: "neuraos-desktop"

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* ═══════════════════════════════════════════════════════════
               LEFT SIDEBAR (220px)
               ═══════════════════════════════════════════════════════════ */
            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: 220
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 0

                    /* Settings title header */
                    Item {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 52

                        Row {
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: 10
                            spacing: 10

                            Components.CanvasIcon {
                                anchors.verticalCenter: parent.verticalCenter
                                iconName: "gear"
                                iconSize: 18
                                iconColor: Theme.primary
                            }

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Settings"
                                color: Theme.text
                                font.pixelSize: 18
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* Thin separator below title */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 1
                        color: Theme.glassBorder
                        Layout.bottomMargin: 8
                    }

                    /* Category list */
                    ListView {
                        id: categoryList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        model: categoryModel
                        spacing: 2
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        delegate: Rectangle {
                            width: categoryList.width
                            height: 40
                            radius: Theme.radiusSmall
                            color: selectedCategory === model.key
                                ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                : catMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }

                            Row {
                                anchors.fill: parent
                                anchors.leftMargin: 12
                                spacing: 12

                                Components.CanvasIcon {
                                    anchors.verticalCenter: parent.verticalCenter
                                    iconName: model.ico
                                    iconSize: 16
                                    iconColor: selectedCategory === model.key ? Theme.primary : Theme.textDim
                                }

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: model.label
                                    font.pixelSize: 13
                                    font.family: Theme.fontFamily
                                    font.bold: selectedCategory === model.key
                                    color: selectedCategory === model.key ? Theme.primary : Theme.text
                                }
                            }

                            /* Selection indicator bar */
                            Rectangle {
                                anchors.left: parent.left
                                anchors.verticalCenter: parent.verticalCenter
                                width: 3
                                height: selectedCategory === model.key ? 20 : 0
                                radius: 2
                                color: Theme.primary
                                Behavior on height { NumberAnimation { duration: Theme.animFast; easing.type: Easing.OutQuad } }
                            }

                            MouseArea {
                                id: catMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: selectedCategory = model.key
                            }
                        }
                    }

                    /* Version info at bottom of sidebar */
                    Text {
                        Layout.fillWidth: true
                        Layout.topMargin: 8
                        horizontalAlignment: Text.AlignHCenter
                        text: "NeuralOS v4.0.0"
                        font.pixelSize: 10
                        font.family: Theme.fontFamily
                        color: Theme.textMuted
                    }
                }
            }

            /* Vertical separator */
            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: 1
                color: Theme.glassBorder
            }

            /* ═══════════════════════════════════════════════════════════
               RIGHT CONTENT AREA
               ═══════════════════════════════════════════════════════════ */
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentHeight: contentColumn.height + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds

                ColumnLayout {
                    id: contentColumn
                    width: parent.width
                    spacing: 0

                    /* ─────────────────────────────────────
                       SYSTEM PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "system"

                        Text {
                            text: "System"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "General system preferences and behavior"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Theme toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Theme"
                            description: "Switch between dark and light mode"

                            Row {
                                spacing: 8
                                anchors.verticalCenter: parent.verticalCenter

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: Settings.theme === "dark" ? "Dark" : "Light"
                                    color: Theme.primary
                                    font.pixelSize: 12
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }

                                SettingToggle {
                                    checked: Settings.theme === "dark"
                                    onToggled: Settings.theme = checked ? "dark" : "light"
                                }
                            }
                        }

                        /* Performance mode dropdown */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Performance Mode"
                            description: "Adjust system resource allocation"

                            Row {
                                spacing: 6
                                anchors.verticalCenter: parent.verticalCenter

                                Repeater {
                                    model: ["Balanced", "Performance", "Power Saver"]

                                    Rectangle {
                                        width: 90; height: 30
                                        radius: Theme.radiusTiny
                                        color: performanceMode === (index === 1)
                                            ? Theme.primary
                                            : (index === 0 && !performanceMode ? Theme.primary : Theme.surfaceAlt)
                                        property bool isActive: (index === 0 && !performanceMode) || (index === 1 && performanceMode)

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData
                                            font.pixelSize: 10
                                            font.family: Theme.fontFamily
                                            color: parent.isActive ? "#000000" : Theme.text
                                        }
                                        MouseArea {
                                            anchors.fill: parent
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: performanceMode = (index === 1)
                                        }
                                    }
                                }
                            }
                        }

                        /* Animation speed */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Animation Speed"
                            description: "Control UI transition speed"

                            SettingSlider {
                                anchors.verticalCenter: parent.verticalCenter
                                width: 200
                                value: animationSpeed
                                minVal: 0.5
                                maxVal: 2.0
                                displayText: animationSpeed.toFixed(1) + "x"
                                sliderColor: Theme.primary
                                onValueChanged: animationSpeed = value
                            }
                        }

                        /* Auto-start toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Auto-Start Dashboard"
                            description: "Launch dashboard on system boot"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: true
                            }
                        }

                        /* Timezone */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Timezone"
                            description: "Current system timezone"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "UTC+0 (Auto)"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       DISPLAY PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "display"

                        Text {
                            text: "Display"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Configure wallpaper, scaling, and resolution"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Wallpaper color picker */
                        Rectangle {
                            Layout.fillWidth: true
                            height: 100
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 16
                                spacing: 10

                                Text {
                                    text: "Wallpaper Color"
                                    color: Theme.text
                                    font.pixelSize: 13
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }

                                Row {
                                    spacing: 14

                                    Repeater {
                                        model: [
                                            { c: "#0D0D12", name: "Midnight" },
                                            { c: "#1a2744", name: "Ocean" },
                                            { c: "#142214", name: "Forest" },
                                            { c: "#2d1b3e", name: "Nebula" },
                                            { c: "#3a1a1a", name: "Crimson" },
                                            { c: "#1a3333", name: "Teal" }
                                        ]

                                        Rectangle {
                                            width: 36; height: 36
                                            radius: 18
                                            color: modelData.c
                                            border.width: selectedWallpaperColor === modelData.c ? 3 : 1
                                            border.color: selectedWallpaperColor === modelData.c
                                                ? Theme.primary
                                                : Theme.glassBorder

                                            Behavior on border.width { NumberAnimation { duration: Theme.animFast } }

                                            /* Checkmark for selected */
                                            Text {
                                                anchors.centerIn: parent
                                                text: "\u2713"
                                                color: "#FFFFFF"
                                                font.pixelSize: 14
                                                font.bold: true
                                                visible: selectedWallpaperColor === modelData.c
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                hoverEnabled: true
                                                onClicked: selectedWallpaperColor = modelData.c
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /* Display Scale slider */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Display Scale"
                            description: "Adjust UI element size (100% - 200%)"

                            SettingSlider {
                                anchors.verticalCenter: parent.verticalCenter
                                width: 220
                                value: displayScale
                                minVal: 100
                                maxVal: 200
                                displayText: Math.round(displayScale) + "%"
                                sliderColor: Theme.primary
                                onValueChanged: displayScale = value
                            }
                        }

                        /* Resolution info */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Resolution"
                            description: "Current display resolution"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "1920 x 1080 @ 60Hz"
                                color: Theme.primary
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Brightness */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Brightness"
                            description: "Screen brightness level"

                            SettingSlider {
                                anchors.verticalCenter: parent.verticalCenter
                                width: 220
                                value: 80
                                minVal: 10
                                maxVal: 100
                                displayText: "80%"
                                sliderColor: Theme.warning
                            }
                        }

                        /* Night Light */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Night Light"
                            description: "Reduce blue light for eye comfort"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: false
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       SOUND PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "sound"

                        Text {
                            text: "Sound"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Audio output, volume, and notification sounds"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Master volume slider */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Master Volume"
                            description: "System-wide audio output level"

                            SettingSlider {
                                anchors.verticalCenter: parent.verticalCenter
                                width: 220
                                value: masterVolume
                                minVal: 0
                                maxVal: 100
                                displayText: Math.round(masterVolume) + "%"
                                sliderColor: Theme.success
                                onValueChanged: masterVolume = value
                            }
                        }

                        /* Notification sounds toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Notification Sounds"
                            description: "Play sound on new notifications"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: notificationSounds
                                onToggled: notificationSounds = checked
                            }
                        }

                        /* System sounds toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "System Sounds"
                            description: "Play sounds for system events (login, errors)"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: systemSounds
                                onToggled: systemSounds = checked
                            }
                        }

                        /* Output device */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Output Device"
                            description: "Current audio output"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Built-in Speakers"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Input device */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Input Device"
                            description: "Current microphone source"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Built-in Microphone"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       NETWORK PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "network"

                        Text {
                            text: "Network"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "WiFi, Ethernet, and proxy configuration"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* WiFi status */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "WiFi"
                            description: "Wireless network connection"

                            Row {
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 8

                                Rectangle {
                                    width: 8; height: 8; radius: 4
                                    color: Theme.success
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    text: "Connected"
                                    color: Theme.success
                                    font.pixelSize: 12
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                        }

                        /* Connected network name */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Network Name (SSID)"
                            description: "Currently connected wireless network"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: connectedNetwork
                                color: Theme.primary
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* IP Address */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "IP Address"
                            description: "Local network address"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: ipAddress
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: "monospace"
                            }
                        }

                        /* Proxy toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Proxy"
                            description: "Route traffic through a proxy server"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: proxyEnabled
                                onToggled: proxyEnabled = checked
                            }
                        }

                        /* DNS */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "DNS Server"
                            description: "Domain name resolution"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "1.1.1.1 (Cloudflare)"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       AI & NPU PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "ainpu"

                        Text {
                            text: "AI & NPU"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Neural Processing Unit configuration and AI inference"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* NPU Status */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "NPU Status"
                            description: "Neural Processing Unit availability"

                            Row {
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 8

                                Rectangle {
                                    width: 8; height: 8; radius: 4
                                    color: NPUMonitor.npuAvailable ? Theme.success : Theme.error
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: NPUMonitor.npuAvailable ? "Available" : "Not Detected"
                                    color: NPUMonitor.npuAvailable ? Theme.success : Theme.error
                                    font.pixelSize: 12
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }
                            }
                        }

                        /* NPU utilization */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "NPU Utilization"
                            description: "Current neural processor load"

                            Row {
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 10

                                Rectangle {
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: 100; height: 6; radius: 3
                                    color: Theme.surfaceLight

                                    Rectangle {
                                        width: parent.width * (NPUMonitor.npuUtilization / 100)
                                        height: parent.height; radius: 3
                                        color: Theme.secondary
                                    }
                                }

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: Math.round(NPUMonitor.npuUtilization) + "%"
                                    color: Theme.secondary
                                    font.pixelSize: 12
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }
                            }
                        }

                        /* Model cache size */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Model Cache Size"
                            description: "Cached AI model data on disk"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: modelCacheSize.toFixed(1) + " GB"
                                color: Theme.text
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Auto-inference toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Auto-Inference"
                            description: "Automatically run inference on compatible tasks"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: autoInference
                                onToggled: autoInference = checked
                            }
                        }

                        /* NPU Power Mode */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "NPU Power Mode"
                            description: "Balance between performance and power usage"

                            Row {
                                spacing: 6
                                anchors.verticalCenter: parent.verticalCenter

                                Repeater {
                                    model: [
                                        { key: "low",       label: "Low Power" },
                                        { key: "balanced",  label: "Balanced" },
                                        { key: "max",       label: "Max Perf" }
                                    ]

                                    Rectangle {
                                        width: 80; height: 28
                                        radius: Theme.radiusTiny
                                        color: npuPowerMode === modelData.key ? Theme.secondary : Theme.surfaceAlt

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData.label
                                            font.pixelSize: 10
                                            font.family: Theme.fontFamily
                                            color: npuPowerMode === modelData.key ? "#FFFFFF" : Theme.text
                                        }

                                        MouseArea {
                                            anchors.fill: parent
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: npuPowerMode = modelData.key
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       PRIVACY PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "privacy"

                        Text {
                            text: "Privacy"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Firewall, telemetry, and lock screen settings"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Firewall status */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Firewall"
                            description: "System network firewall"

                            Row {
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 10

                                Row {
                                    anchors.verticalCenter: parent.verticalCenter
                                    spacing: 6

                                    Rectangle {
                                        width: 8; height: 8; radius: 4
                                        anchors.verticalCenter: parent.verticalCenter
                                        color: firewallEnabled ? Theme.success : Theme.error
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: firewallEnabled ? "Active" : "Disabled"
                                        color: firewallEnabled ? Theme.success : Theme.error
                                        font.pixelSize: 11
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                    }
                                }

                                SettingToggle {
                                    anchors.verticalCenter: parent.verticalCenter
                                    checked: firewallEnabled
                                    onToggled: firewallEnabled = checked
                                }
                            }
                        }

                        /* Telemetry toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Telemetry"
                            description: "Send anonymous usage statistics"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: telemetryEnabled
                                onToggled: telemetryEnabled = checked
                            }
                        }

                        /* Screen lock toggle */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Screen Lock"
                            description: "Lock screen after inactivity"

                            SettingToggle {
                                anchors.verticalCenter: parent.verticalCenter
                                checked: screenLockEnabled
                                onToggled: screenLockEnabled = checked
                            }
                        }

                        /* Encryption */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Disk Encryption"
                            description: "Full-disk encryption status"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "AES-256-XTS"
                                color: Theme.success
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Secure boot */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Secure Boot"
                            description: "UEFI verified boot chain"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Enabled"
                                color: Theme.success
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       APPS PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "apps"

                        Text {
                            text: "Apps"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Default applications and preferences"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        Repeater {
                            model: [
                                { role: "Browser",  app: "NeuralBrowse",    icon: "globe" },
                                { role: "Terminal", app: "NeuraTerm",       icon: "terminal" },
                                { role: "Files",    app: "FileManager",     icon: "folder" },
                                { role: "Editor",   app: "NeuraEdit",       icon: "edit" }
                            ]

                            Rectangle {
                                Layout.fillWidth: true
                                height: 60
                                radius: Theme.radiusSmall
                                color: Theme.surface

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 16
                                    anchors.rightMargin: 16
                                    spacing: 14

                                    Rectangle {
                                        width: 36; height: 36; radius: 10
                                        color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.12)

                                        Components.CanvasIcon {
                                            anchors.centerIn: parent
                                            iconName: modelData.icon
                                            iconSize: 18
                                            iconColor: Theme.primary
                                        }
                                    }

                                    Column {
                                        Layout.fillWidth: true
                                        spacing: 2

                                        Text {
                                            text: "Default " + modelData.role
                                            color: Theme.text
                                            font.pixelSize: 12
                                            font.bold: true
                                            font.family: Theme.fontFamily
                                        }
                                        Text {
                                            text: modelData.app
                                            color: Theme.textDim
                                            font.pixelSize: 11
                                            font.family: Theme.fontFamily
                                        }
                                    }

                                    Components.CanvasIcon {
                                        iconName: "chevron-right"
                                        iconSize: 14
                                        iconColor: Theme.textMuted
                                    }
                                }
                            }
                        }

                        /* Startup apps */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Startup Applications"
                            description: "Apps that launch automatically on login"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "3 apps"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       STORAGE PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "storage"

                        Text {
                            text: "Storage"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "Disk usage, cache, and temporary file management"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Disk usage bar */
                        Rectangle {
                            Layout.fillWidth: true
                            height: 100
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 16
                                spacing: 10

                                RowLayout {
                                    Layout.fillWidth: true

                                    Text {
                                        text: "Disk Usage"
                                        color: Theme.text
                                        font.pixelSize: 13
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                    }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        text: SystemInfo.diskUsed + " / " + SystemInfo.diskTotal + " GB"
                                        color: Theme.textDim
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                    }
                                }

                                /* Progress bar */
                                Rectangle {
                                    Layout.fillWidth: true
                                    height: 10
                                    radius: 5
                                    color: Theme.surfaceLight

                                    Rectangle {
                                        property real ratio: {
                                            var total = parseFloat(SystemInfo.diskTotal)
                                            var used = parseFloat(SystemInfo.diskUsed)
                                            if (total <= 0) return 0
                                            return Math.min(1.0, used / total)
                                        }
                                        width: parent.width * ratio
                                        height: parent.height
                                        radius: 5
                                        color: ratio > 0.85 ? Theme.error : ratio > 0.7 ? Theme.warning : Theme.primary

                                        Behavior on width { NumberAnimation { duration: Theme.animFast } }
                                    }
                                }

                                Text {
                                    property real ratio: {
                                        var total = parseFloat(SystemInfo.diskTotal)
                                        var used = parseFloat(SystemInfo.diskUsed)
                                        if (total <= 0) return 0
                                        return Math.min(100, (used / total) * 100)
                                    }
                                    text: Math.round(ratio) + "% used"
                                    color: Theme.textMuted
                                    font.pixelSize: 11
                                    font.family: Theme.fontFamily
                                }
                            }
                        }

                        /* Cache size */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Cache Size"
                            description: "Application and system cache"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: cacheSize.toFixed(1) + " GB"
                                color: Theme.warning
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Tmp cleanup button */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Temporary Files"
                            description: "Clean up /tmp and transient data"

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                width: cleanLabel.implicitWidth + 24
                                height: 30
                                radius: Theme.radiusTiny
                                color: cleanMa.containsMouse ? Qt.darker(Theme.error, 1.1) : Theme.error

                                Text {
                                    id: cleanLabel
                                    anchors.centerIn: parent
                                    text: "Clean Up"
                                    color: "#FFFFFF"
                                    font.pixelSize: 11
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }

                                MouseArea {
                                    id: cleanMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: cacheSize = 0.0
                                }
                            }
                        }

                        /* Memory info */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Total Memory"
                            description: "Installed system RAM"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: SystemInfo.memTotal + " MB"
                                color: Theme.text
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       USERS PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "users"

                        Text {
                            text: "Users"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "User accounts and hostname"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* Current user card */
                        Rectangle {
                            Layout.fillWidth: true
                            height: 110
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 20
                                spacing: 18

                                /* User avatar */
                                Rectangle {
                                    width: 64; height: 64; radius: 32
                                    color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)

                                    Text {
                                        anchors.centerIn: parent
                                        text: "R"
                                        font.pixelSize: 28
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                        color: Theme.primary
                                    }
                                }

                                Column {
                                    Layout.fillWidth: true
                                    spacing: 4

                                    Text {
                                        text: "root"
                                        color: Theme.text
                                        font.pixelSize: 18
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "Administrator  |  UID 0"
                                        color: Theme.textDim
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "Shell: /bin/bash  |  Home: /root"
                                        color: Theme.textMuted
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                    }
                                }

                                /* Status badge */
                                Rectangle {
                                    width: statusText.implicitWidth + 16
                                    height: 24
                                    radius: 12
                                    color: Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.15)
                                    Layout.alignment: Qt.AlignVCenter

                                    Text {
                                        id: statusText
                                        anchors.centerIn: parent
                                        text: "Active"
                                        color: Theme.success
                                        font.pixelSize: 10
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                    }
                                }
                            }
                        }

                        /* Hostname */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Hostname"
                            description: "Machine network identifier"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: currentHostname
                                color: Theme.primary
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Groups */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Groups"
                            description: "User group memberships"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "root, sudo, docker, kvm"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Last login */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Last Login"
                            description: "Most recent session"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Today at boot"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* ─────────────────────────────────────
                       ABOUT PAGE
                       ───────────────────────────────────── */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 24
                        spacing: 12
                        visible: selectedCategory === "about"

                        Text {
                            text: "About"
                            color: Theme.text
                            font.pixelSize: 22
                            font.bold: true
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 4
                        }
                        Text {
                            text: "System information and version details"
                            color: Theme.textDim
                            font.pixelSize: 12
                            font.family: Theme.fontFamily
                            Layout.bottomMargin: 8
                        }

                        /* NeuralOS branding card */
                        Rectangle {
                            Layout.fillWidth: true
                            height: 140
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            Column {
                                anchors.centerIn: parent
                                spacing: 8

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "N"
                                    font.pixelSize: 44
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                    color: Theme.primary
                                }

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "NeuralOS"
                                    color: Theme.text
                                    font.pixelSize: 20
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                }

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "AI-Native Desktop Operating System"
                                    color: Theme.textDim
                                    font.pixelSize: 12
                                    font.family: Theme.fontFamily
                                }
                            }
                        }

                        /* Version */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Version"
                            description: "Operating system version"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "v4.0.0"
                                color: Theme.primary
                                font.pixelSize: 13
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Build info */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Build"
                            description: "Build identifier and date"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "2026.02.07-stable"
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Qt version */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Qt Version"
                            description: "UI framework version"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Qt 5.15.2"
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Kernel */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Kernel"
                            description: "Linux kernel version"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "6.8.0-neuraos"
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* CPU */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Processor"
                            description: "CPU model and core count"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: SystemInfo.cpuModel + " (" + SystemInfo.cpuCores + " cores)"
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* Memory */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "Memory"
                            description: "Installed system RAM"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: SystemInfo.memTotal + " MB"
                                color: Theme.text
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }

                        /* NPU */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "NPU"
                            description: "Neural Processing Unit"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: NPUMonitor.npuAvailable ? "Detected" : "Not Available"
                                color: NPUMonitor.npuAvailable ? Theme.success : Theme.textMuted
                                font.pixelSize: 12
                                font.bold: true
                                font.family: Theme.fontFamily
                            }
                        }

                        /* License */
                        SettingCard {
                            Layout.fillWidth: true
                            label: "License"
                            description: "Software license"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "GPL-3.0"
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                            }
                        }
                    }

                    /* Bottom spacer */
                    Item { Layout.fillWidth: true; height: 20 }
                }
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════════
       INLINE COMPONENTS
       ═══════════════════════════════════════════════════════════════════ */

    /* ─── Setting Card: a standard row with label on left, control slot on right ─── */
    component SettingCard: Rectangle {
        id: card
        height: 60
        radius: Theme.radiusSmall
        color: Theme.surface

        property string label: ""
        property string description: ""
        default property alias control: controlSlot.children

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12

            Column {
                Layout.fillWidth: true
                spacing: 2

                Text {
                    text: card.label
                    color: Theme.text
                    font.pixelSize: 13
                    font.family: Theme.fontFamily
                }
                Text {
                    text: card.description
                    color: Theme.textDim
                    font.pixelSize: 11
                    font.family: Theme.fontFamily
                    visible: card.description !== ""
                }
            }

            Item {
                id: controlSlot
                Layout.preferredWidth: childrenRect.width
                Layout.preferredHeight: childrenRect.height
            }
        }
    }

    /* ─── Toggle Switch: Rectangle with sliding circle (animated) ─── */
    component SettingToggle: Item {
        id: toggleRoot
        width: 44; height: 24

        property bool checked: false
        signal toggled(bool checked)

        Rectangle {
            anchors.fill: parent
            radius: height / 2
            color: toggleRoot.checked ? Theme.primary : Theme.surfaceLight

            Behavior on color { ColorAnimation { duration: Theme.animFast } }

            Rectangle {
                width: 18; height: 18
                radius: 9
                anchors.verticalCenter: parent.verticalCenter
                x: toggleRoot.checked ? parent.width - width - 3 : 3
                color: "#FFFFFF"

                Behavior on x { NumberAnimation { duration: Theme.animFast; easing.type: Easing.OutQuad } }
            }
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.PointingHandCursor
            onClicked: {
                toggleRoot.checked = !toggleRoot.checked
                toggleRoot.toggled(toggleRoot.checked)
            }
        }
    }

    /* ─── Slider: Rectangle track with draggable handle ─── */
    component SettingSlider: Item {
        id: sliderRoot
        height: 24

        property real value: 50
        property real minVal: 0
        property real maxVal: 100
        property string displayText: ""
        property color sliderColor: Theme.primary

        Row {
            anchors.fill: parent
            spacing: 10

            Item {
                width: sliderRoot.width - valueLabel.width - 10
                height: parent.height

                /* Track */
                Rectangle {
                    anchors.verticalCenter: parent.verticalCenter
                    width: parent.width
                    height: 4
                    radius: 2
                    color: Theme.surfaceLight

                    /* Filled portion */
                    Rectangle {
                        width: parent.width * ((sliderRoot.value - sliderRoot.minVal)
                               / (sliderRoot.maxVal - sliderRoot.minVal))
                        height: parent.height
                        radius: 2
                        color: sliderRoot.sliderColor
                    }
                }

                /* Draggable handle */
                Rectangle {
                    x: Math.max(0, Math.min(parent.width - width,
                       (sliderRoot.value - sliderRoot.minVal)
                       / (sliderRoot.maxVal - sliderRoot.minVal) * parent.width - width / 2))
                    anchors.verticalCenter: parent.verticalCenter
                    width: 16; height: 16; radius: 8
                    color: sliderRoot.sliderColor
                    border.width: 2
                    border.color: Qt.lighter(sliderRoot.sliderColor, 1.3)

                    Behavior on x { NumberAnimation { duration: 50 } }
                }

                MouseArea {
                    anchors.fill: parent
                    onPressed: updateSlider(mouse)
                    onPositionChanged: if (pressed) updateSlider(mouse)

                    function updateSlider(m) {
                        var ratio = Math.max(0, Math.min(1, m.x / width))
                        sliderRoot.value = sliderRoot.minVal + ratio * (sliderRoot.maxVal - sliderRoot.minVal)
                    }
                }
            }

            Text {
                id: valueLabel
                anchors.verticalCenter: parent.verticalCenter
                text: sliderRoot.displayText
                color: sliderRoot.sliderColor
                font.pixelSize: 12
                font.bold: true
                font.family: Theme.fontFamily
                width: 50
                horizontalAlignment: Text.AlignRight
            }
        }
    }
}
