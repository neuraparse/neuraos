import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: settingsPage

    property int selectedCategory: 0

    Flickable {
        anchors.fill: parent
        anchors.margins: 20
        contentHeight: col.height
        clip: true

        ColumnLayout {
            id: col
            width: parent.width
            spacing: 16

            Components.SectionHeader { title: "Settings" }

            /* Category tabs */
            Rectangle {
                Layout.fillWidth: true
                height: 44
                radius: Theme.radius
                color: Theme.surface

                Row {
                    anchors.centerIn: parent
                    spacing: 4

                    Repeater {
                        model: ListModel {
                            ListElement { label: "System";  idx: 0 }
                            ListElement { label: "Display"; idx: 1 }
                            ListElement { label: "AI Runtime"; idx: 2 }
                            ListElement { label: "Network"; idx: 3 }
                            ListElement { label: "About"; idx: 4 }
                        }

                        delegate: Rectangle {
                            width: 100; height: 34
                            radius: Theme.radiusSmall
                            color: selectedCategory === model.idx ?
                                Theme.primary : (catMa.containsMouse ? Theme.surfaceAlt : "transparent")

                            Text {
                                anchors.centerIn: parent
                                text: model.label
                                color: selectedCategory === model.idx ? "#000000" : Theme.text
                                font.pixelSize: Theme.fontSizeSmall
                                font.bold: selectedCategory === model.idx
                            }

                            MouseArea {
                                id: catMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: selectedCategory = model.idx
                            }
                        }
                    }
                }
            }

            /* System Settings */
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12
                visible: selectedCategory === 0

                /* Hostname */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Hostname"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "Device network name"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Text {
                            text: SystemInfo.hostname
                            color: Theme.primary
                            font.pixelSize: Theme.fontSizeNormal
                            font.bold: true
                        }
                    }
                }

                /* Timezone */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Timezone"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "System time zone"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Text {
                            text: Settings.timezone
                            color: Theme.text
                            font.pixelSize: Theme.fontSizeNormal
                        }
                    }
                }

                /* Auto-start dashboard */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Auto-start Dashboard"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "Launch dashboard on boot"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Components.ToggleSwitch {
                            checked: Settings.autoStart
                            onToggled: Settings.autoStart = checked
                        }
                    }
                }
            }

            /* Display Settings */
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12
                visible: selectedCategory === 1

                /* Brightness */
                Rectangle {
                    Layout.fillWidth: true
                    height: 80
                    radius: Theme.radius
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 8

                        RowLayout {
                            Text { text: "Brightness"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: Math.round(brightnessSlider.value) + "%"
                                color: Theme.primary
                                font.pixelSize: Theme.fontSizeNormal
                                font.bold: true
                            }
                        }

                        /* Custom slider */
                        Item {
                            Layout.fillWidth: true
                            height: 24

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width
                                height: 4
                                radius: 2
                                color: Theme.surfaceLight

                                Rectangle {
                                    width: parent.width * brightnessSlider.value / 100
                                    height: parent.height
                                    radius: 2
                                    color: Theme.primary
                                }
                            }

                            MouseArea {
                                id: brightnessSlider
                                anchors.fill: parent
                                property real value: Settings.brightness

                                onPressed: {
                                    value = Math.max(10, Math.min(100, mouse.x / width * 100))
                                    Settings.brightness = value
                                }
                                onPositionChanged: {
                                    if (pressed) {
                                        value = Math.max(10, Math.min(100, mouse.x / width * 100))
                                        Settings.brightness = value
                                    }
                                }
                            }

                            /* Slider knob */
                            Rectangle {
                                x: brightnessSlider.value / 100 * parent.width - 8
                                anchors.verticalCenter: parent.verticalCenter
                                width: 16; height: 16; radius: 8
                                color: Theme.primary
                            }
                        }
                    }
                }

                /* Volume */
                Rectangle {
                    Layout.fillWidth: true
                    height: 80
                    radius: Theme.radius
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 8

                        RowLayout {
                            Text { text: "Volume"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: Math.round(volumeSlider.value) + "%"
                                color: Theme.success
                                font.pixelSize: Theme.fontSizeNormal
                                font.bold: true
                            }
                        }

                        Item {
                            Layout.fillWidth: true
                            height: 24

                            Rectangle {
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width
                                height: 4
                                radius: 2
                                color: Theme.surfaceLight

                                Rectangle {
                                    width: parent.width * volumeSlider.value / 100
                                    height: parent.height
                                    radius: 2
                                    color: Theme.success
                                }
                            }

                            MouseArea {
                                id: volumeSlider
                                anchors.fill: parent
                                property real value: Settings.volume

                                onPressed: {
                                    value = Math.max(0, Math.min(100, mouse.x / width * 100))
                                    Settings.volume = value
                                }
                                onPositionChanged: {
                                    if (pressed) {
                                        value = Math.max(0, Math.min(100, mouse.x / width * 100))
                                        Settings.volume = value
                                    }
                                }
                            }

                            Rectangle {
                                x: volumeSlider.value / 100 * parent.width - 8
                                anchors.verticalCenter: parent.verticalCenter
                                width: 16; height: 16; radius: 8
                                color: Theme.success
                            }
                        }
                    }
                }

                /* Theme */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Theme"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "UI color scheme"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Row {
                            spacing: 8

                            Repeater {
                                model: ["dark", "light", "auto"]

                                Rectangle {
                                    width: 64; height: 30
                                    radius: Theme.radiusSmall
                                    color: Settings.theme === modelData ? Theme.primary : Theme.surfaceAlt
                                    border.width: 1
                                    border.color: Settings.theme === modelData ? Theme.primary : Theme.surfaceLight

                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData.charAt(0).toUpperCase() + modelData.slice(1)
                                        color: Settings.theme === modelData ? "#000000" : Theme.text
                                        font.pixelSize: 11
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: Settings.theme = modelData
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* AI Runtime Settings */
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12
                visible: selectedCategory === 2

                /* AI Backend */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "AI Backend"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "Default inference engine"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Row {
                            spacing: 6

                            Repeater {
                                model: NPIE.backends

                                Rectangle {
                                    width: 80; height: 30
                                    radius: Theme.radiusSmall
                                    color: NPIE.currentBackend === modelData ? Theme.primary : Theme.surfaceAlt

                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData
                                        color: NPIE.currentBackend === modelData ? "#000000" : Theme.text
                                        font.pixelSize: 11
                                        font.bold: NPIE.currentBackend === modelData
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: NPIE.setBackend(modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                /* NPIE Info */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "NPIE Version"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: "NeuraParse Inference Engine"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Components.StatusBadge {
                            text: NPIE.version
                            badgeColor: Theme.primary
                        }
                    }
                }

                /* NPU Power */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Column {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "NPU Power Management"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Text { text: NPUMonitor.deviceCount + " device(s) | " + NPUMonitor.powerMw + " mW"; color: Theme.textDim; font.pixelSize: Theme.fontSizeSmall }
                        }

                        Components.ToggleSwitch {
                            checked: true
                            onToggled: NPUMonitor.setPower(checked)
                        }
                    }
                }

                /* NPU Frequency */
                Rectangle {
                    Layout.fillWidth: true
                    height: 80
                    radius: Theme.radius
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 8

                        RowLayout {
                            Text { text: "NPU Frequency"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: NPUMonitor.frequencyMhz + " MHz"
                                color: Theme.secondary
                                font.pixelSize: Theme.fontSizeNormal
                                font.bold: true
                            }
                        }

                        Row {
                            spacing: 8

                            Repeater {
                                model: [200, 500, 800, 1000]

                                Rectangle {
                                    width: 70; height: 28
                                    radius: Theme.radiusSmall
                                    color: NPUMonitor.frequencyMhz === modelData ? Theme.secondary : Theme.surfaceAlt

                                    Text {
                                        anchors.centerIn: parent
                                        text: modelData + " MHz"
                                        color: NPUMonitor.frequencyMhz === modelData ? "#FFFFFF" : Theme.text
                                        font.pixelSize: 10
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: NPUMonitor.setFrequency(modelData)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* Network Settings */
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12
                visible: selectedCategory === 3

                Repeater {
                    model: NetworkManager.interfaces

                    Rectangle {
                        Layout.fillWidth: true
                        height: 70
                        radius: Theme.radius
                        color: Theme.surface

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 16
                            spacing: 16

                            Rectangle {
                                width: 40; height: 40; radius: 20
                                color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                Text {
                                    anchors.centerIn: parent
                                    text: "\u2301"
                                    font.pixelSize: 20
                                    color: Theme.primary
                                }
                            }

                            Column {
                                Layout.fillWidth: true
                                spacing: 2
                                Text {
                                    text: modelData.name || "Unknown"
                                    color: Theme.text
                                    font.pixelSize: Theme.fontSizeNormal
                                    font.bold: true
                                }
                                Text {
                                    text: (modelData.address || "No address") + " | " +
                                          (modelData.type || "Unknown type")
                                    color: Theme.textDim
                                    font.pixelSize: Theme.fontSizeSmall
                                }
                            }

                            Components.StatusBadge {
                                text: modelData.isUp ? "Up" : "Down"
                                badgeColor: modelData.isUp ? Theme.success : Theme.error
                            }
                        }
                    }
                }

                /* Hostname */
                Rectangle {
                    Layout.fillWidth: true
                    height: 60
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Text { text: "Network Hostname"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: NetworkManager.hostname
                            color: Theme.primary
                            font.pixelSize: Theme.fontSizeNormal
                            font.bold: true
                        }
                    }
                }
            }

            /* About */
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12
                visible: selectedCategory === 4

                /* Logo */
                Rectangle {
                    Layout.fillWidth: true
                    height: 140
                    radius: Theme.radius
                    color: Theme.surface

                    Column {
                        anchors.centerIn: parent
                        spacing: 12

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "N"
                            font.pixelSize: 48
                            font.bold: true
                            color: Theme.primary
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "NeuralOS"
                            color: Theme.text
                            font.pixelSize: Theme.fontSizeLarge
                            font.bold: true
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "AI-Native Embedded Linux v2.0.0"
                            color: Theme.textDim
                            font.pixelSize: Theme.fontSizeSmall
                        }
                    }
                }

                /* Info list */
                Repeater {
                    model: ListModel {
                        ListElement { label: "Kernel";    prop: "kernelVersion" }
                        ListElement { label: "Hostname";  prop: "hostname" }
                        ListElement { label: "Uptime";    prop: "uptime" }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 48
                        radius: Theme.radius
                        color: Theme.surface

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 16

                            Text {
                                text: model.label
                                color: Theme.textDim
                                font.pixelSize: Theme.fontSizeNormal
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: model.prop === "kernelVersion" ? SystemInfo.kernelVersion :
                                      model.prop === "hostname" ? SystemInfo.hostname :
                                      SystemInfo.uptime
                                color: Theme.text
                                font.pixelSize: Theme.fontSizeNormal
                            }
                        }
                    }
                }

                /* NPIE info */
                Rectangle {
                    Layout.fillWidth: true
                    height: 48
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Text { text: "NPIE Version"; color: Theme.textDim; font.pixelSize: Theme.fontSizeNormal }
                        Item { Layout.fillWidth: true }
                        Text { text: NPIE.version; color: Theme.primary; font.pixelSize: Theme.fontSizeNormal; font.bold: true }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 48
                    radius: Theme.radius
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 16

                        Text { text: "NPU Devices"; color: Theme.textDim; font.pixelSize: Theme.fontSizeNormal }
                        Item { Layout.fillWidth: true }
                        Text { text: NPUMonitor.deviceCount + " detected"; color: Theme.text; font.pixelSize: Theme.fontSizeNormal }
                    }
                }

                /* Save / Reset */
                RowLayout {
                    Layout.alignment: Qt.AlignRight
                    spacing: 12

                    Rectangle {
                        width: 100; height: 36
                        radius: Theme.radiusSmall
                        color: Theme.surfaceAlt

                        Text {
                            anchors.centerIn: parent
                            text: "Reset"
                            color: Theme.warning
                            font.pixelSize: Theme.fontSizeSmall
                        }

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: Settings.reset()
                        }
                    }

                    Rectangle {
                        width: 100; height: 36
                        radius: Theme.radiusSmall
                        color: Theme.primary

                        Text {
                            anchors.centerIn: parent
                            text: "Save"
                            color: "#000000"
                            font.bold: true
                            font.pixelSize: Theme.fontSizeSmall
                        }

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: Settings.save()
                        }
                    }
                }
            }

            Item { height: 20 }
        }
    }
}
