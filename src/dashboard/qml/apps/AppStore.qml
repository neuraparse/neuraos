import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: storeApp
    anchors.fill: parent

    property int selectedTab: 0
    property string searchText: ""
    property string selectedCat: ""

    property var featuredApps: [
        { name: "TensorFlow Lite", ver: "2.16.0", desc: "Lightweight ML inference engine for edge devices", cat: "AI & ML", ico: "neural", clr: "#FF6F00", size: "45 MB", installed: true },
        { name: "ONNX Runtime", ver: "1.18.0", desc: "Cross-platform ML inference optimizer", cat: "AI & ML", ico: "chip", clr: "#0078D4", size: "62 MB", installed: false },
        { name: "Whisper CPP", ver: "1.6.2", desc: "Automatic speech recognition on device", cat: "AI & ML", ico: "mic", clr: "#10B981", size: "38 MB", installed: false },
        { name: "Stable Diffusion", ver: "2.1", desc: "AI image generation model", cat: "AI & ML", ico: "image", clr: "#A855F7", size: "4.2 GB", installed: false }
    ]

    property var allApps: [
        { name: "Vim", ver: "9.1", desc: "The ubiquitous text editor", cat: "Editors", ico: "edit", clr: "#019733", size: "12 MB", installed: true },
        { name: "Nano", ver: "7.2", desc: "Simple terminal text editor", cat: "Editors", ico: "file", clr: "#4285F4", size: "2 MB", installed: true },
        { name: "Python 3", ver: "3.12", desc: "Python programming language", cat: "Development", ico: "code", clr: "#3776AB", size: "85 MB", installed: true },
        { name: "Node.js", ver: "22.0", desc: "JavaScript runtime", cat: "Development", ico: "code", clr: "#339933", size: "52 MB", installed: false },
        { name: "Docker Engine", ver: "26.1", desc: "Container runtime", cat: "System", ico: "box", clr: "#2496ED", size: "120 MB", installed: false },
        { name: "Wireshark", ver: "4.4", desc: "Network protocol analyzer", cat: "Network", ico: "wifi", clr: "#1679A7", size: "48 MB", installed: false },
        { name: "GIMP", ver: "3.0", desc: "GNU Image Manipulation Program", cat: "Graphics", ico: "image", clr: "#5C5543", size: "95 MB", installed: false },
        { name: "FFmpeg", ver: "7.0", desc: "Multimedia framework for audio/video", cat: "Media", ico: "play", clr: "#007808", size: "28 MB", installed: true },
        { name: "OpenSSH", ver: "9.7", desc: "Secure Shell connectivity tools", cat: "Network", ico: "lock", clr: "#EE0000", size: "4 MB", installed: true },
        { name: "Nmap", ver: "7.95", desc: "Network discovery & security audit", cat: "Security", ico: "shield", clr: "#4682B4", size: "18 MB", installed: false },
        { name: "Git", ver: "2.45", desc: "Distributed version control system", cat: "Development", ico: "refresh", clr: "#F05032", size: "15 MB", installed: true },
        { name: "CMake", ver: "3.29", desc: "Cross-platform build system", cat: "Development", ico: "gear", clr: "#064F8C", size: "22 MB", installed: true },
        { name: "Nginx", ver: "1.26", desc: "High-performance web server", cat: "Server", ico: "server", clr: "#009639", size: "8 MB", installed: false },
        { name: "PostgreSQL", ver: "16.3", desc: "Advanced open source database", cat: "Database", ico: "database", clr: "#336791", size: "145 MB", installed: false },
        { name: "Redis", ver: "7.4", desc: "In-memory data structure store", cat: "Database", ico: "database", clr: "#DC382D", size: "12 MB", installed: false },
        { name: "Mosquitto", ver: "2.0", desc: "MQTT message broker for IoT", cat: "IoT", ico: "globe", clr: "#3C5280", size: "5 MB", installed: false },
        { name: "Kubernetes", ver: "1.30", desc: "Container orchestration platform", cat: "System", ico: "cloud", clr: "#326CE5", size: "380 MB", installed: false },
        { name: "Grafana", ver: "11.0", desc: "Analytics & monitoring dashboard", cat: "System", ico: "dashboard", clr: "#F46800", size: "85 MB", installed: false },
        { name: "Rust", ver: "1.78", desc: "Systems programming language", cat: "Development", ico: "code", clr: "#CE412B", size: "320 MB", installed: false },
        { name: "Go", ver: "1.22", desc: "Fast compiled programming language", cat: "Development", ico: "code", clr: "#00ADD8", size: "180 MB", installed: false },
        { name: "Blender", ver: "4.1", desc: "3D creation suite", cat: "Graphics", ico: "image", clr: "#E87D0D", size: "650 MB", installed: false },
        { name: "LibreOffice", ver: "24.2", desc: "Free office productivity suite", cat: "Office", ico: "file", clr: "#18A303", size: "420 MB", installed: false },
        { name: "Prometheus", ver: "2.52", desc: "Systems monitoring & alerting", cat: "System", ico: "monitor", clr: "#E6522C", size: "45 MB", installed: false },
        { name: "Ansible", ver: "9.5", desc: "IT automation platform", cat: "System", ico: "gear", clr: "#EE0000", size: "55 MB", installed: false },
        { name: "Elasticsearch", ver: "8.14", desc: "Distributed search & analytics", cat: "Database", ico: "search", clr: "#005571", size: "510 MB", installed: false },
        { name: "VLC", ver: "3.0.20", desc: "Multimedia player & framework", cat: "Media", ico: "play", clr: "#FF8800", size: "42 MB", installed: false }
    ]

    function getDisplayApps() {
        var source = selectedTab === 0 ? allApps : allApps.filter(function(a) { return a.installed })
        if (selectedCat !== "") {
            source = source.filter(function(a) { return a.cat === selectedCat })
        }
        if (searchText !== "") {
            source = source.filter(function(a) {
                return a.name.toLowerCase().indexOf(searchText.toLowerCase()) !== -1 ||
                       a.desc.toLowerCase().indexOf(searchText.toLowerCase()) !== -1
            })
        }
        return source
    }

    function getAppCategories() {
        var cats = []
        var seen = {}
        for (var i = 0; i < allApps.length; i++) {
            var c = allApps[i].cat
            if (!seen[c]) { seen[c] = true; cats.push(c) }
        }
        return cats
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Header toolbar */
            Rectangle {
                Layout.fillWidth: true
                height: 56
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    spacing: 12

                    Components.CanvasIcon {
                        iconName: "apps"
                        iconColor: Theme.primary
                        iconSize: 22
                    }

                    Text {
                        text: "App Store"
                        font.pixelSize: 16
                        font.bold: true
                        font.family: Theme.fontFamily
                        color: Theme.text
                    }

                    /* Tabs */
                    Row {
                        spacing: 4
                        Layout.leftMargin: 12

                        Repeater {
                            model: ["Browse", "Installed", "Updates"]

                            Rectangle {
                                width: tabLbl.implicitWidth + 20; height: 28; radius: 14
                                color: selectedTab === index ? Theme.primary : (tabMa.containsMouse ? Theme.surfaceAlt : "transparent")
                                Behavior on color { ColorAnimation { duration: 100 } }

                                Text {
                                    id: tabLbl; anchors.centerIn: parent
                                    text: modelData; font.pixelSize: 11; font.family: Theme.fontFamily
                                    color: selectedTab === index ? "#FFFFFF" : Theme.textDim
                                }
                                MouseArea {
                                    id: tabMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedTab = index
                                }
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Search */
                    Rectangle {
                        width: 200; height: 32; radius: 16
                        color: Theme.surfaceAlt

                        Row {
                            anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 6
                            Components.CanvasIcon { anchors.verticalCenter: parent.verticalCenter; iconName: "search"; iconSize: 14; iconColor: Theme.textDim }
                            TextInput {
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width - 30
                                color: Theme.text; font.pixelSize: 12; font.family: Theme.fontFamily
                                clip: true; selectByMouse: true
                                onTextChanged: storeApp.searchText = text
                                Text { anchors.verticalCenter: parent.verticalCenter; text: "Search packages..."; color: Theme.textMuted; font.pixelSize: 12; visible: !parent.text && !parent.activeFocus }
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Content */
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentHeight: contentCol.height
                clip: true

                ColumnLayout {
                    id: contentCol
                    width: parent.width
                    spacing: 0

                    /* Featured section (only on Browse tab) */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 16
                        spacing: 10
                        visible: selectedTab === 0 && searchText === ""

                        Text {
                            text: "Featured AI & ML"
                            font.pixelSize: 14; font.bold: true; font.family: Theme.fontFamily
                            color: Theme.text
                        }

                        Row {
                            spacing: 8

                            Repeater {
                                model: featuredApps

                                Rectangle {
                                    width: (storeApp.width - 56) / 4
                                    height: 140
                                    radius: Theme.radius
                                    color: featMa.containsMouse ? Theme.surfaceAlt : Theme.surface
                                    Behavior on color { ColorAnimation { duration: 100 } }

                                    ColumnLayout {
                                        anchors.fill: parent; anchors.margins: 12; spacing: 6

                                        RowLayout {
                                            spacing: 8
                                            Rectangle {
                                                width: 36; height: 36; radius: 8
                                                color: Qt.rgba(0.3, 0.3, 0.5, 0.15)
                                                Components.CanvasIcon {
                                                    anchors.centerIn: parent
                                                    iconName: modelData.ico; iconColor: modelData.clr; iconSize: 18
                                                }
                                            }
                                            Column {
                                                spacing: 1
                                                Text { text: modelData.name; font.pixelSize: 12; font.bold: true; font.family: Theme.fontFamily; color: Theme.text }
                                                Text { text: modelData.ver; font.pixelSize: 9; font.family: Theme.fontFamily; color: Theme.textDim }
                                            }
                                        }

                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.desc
                                            font.pixelSize: 10; font.family: Theme.fontFamily
                                            color: Theme.textDim; wrapMode: Text.WordWrap
                                            maximumLineCount: 2; elide: Text.ElideRight
                                        }

                                        Item { Layout.fillHeight: true }

                                        Rectangle {
                                            Layout.fillWidth: true; height: 26; radius: 13
                                            color: modelData.installed ? Theme.surfaceLight : Theme.primary
                                            Text {
                                                anchors.centerIn: parent
                                                text: modelData.installed ? "Installed" : "Install"
                                                font.pixelSize: 10; font.bold: true; font.family: Theme.fontFamily
                                                color: modelData.installed ? Theme.textDim : "#FFFFFF"
                                            }
                                        }
                                    }

                                    MouseArea { id: featMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                                }
                            }
                        }
                    }

                    /* Separator */
                    Rectangle {
                        Layout.fillWidth: true; Layout.leftMargin: 16; Layout.rightMargin: 16
                        height: 1; color: Theme.surfaceLight
                        visible: selectedTab === 0 && searchText === ""
                    }

                    /* Category Filter */
                    Flow {
                        Layout.fillWidth: true
                        Layout.leftMargin: 16; Layout.rightMargin: 16
                        Layout.topMargin: 8
                        spacing: 6
                        visible: selectedTab === 0

                        Rectangle {
                            width: allCatLbl.implicitWidth + 16; height: 26; radius: 13
                            color: selectedCat === "" ? Theme.primary : (allCatMa.containsMouse ? Theme.surfaceAlt : Theme.surface)

                            Text { id: allCatLbl; anchors.centerIn: parent; text: "All"; font.pixelSize: 11; color: selectedCat === "" ? "#FFFFFF" : Theme.text }
                            MouseArea { id: allCatMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: selectedCat = "" }
                        }

                        Repeater {
                            model: getAppCategories()

                            Rectangle {
                                width: acLbl.implicitWidth + 16; height: 26; radius: 13
                                color: selectedCat === modelData ? Theme.primary : (acMa.containsMouse ? Theme.surfaceAlt : Theme.surface)

                                Text { id: acLbl; anchors.centerIn: parent; text: modelData; font.pixelSize: 11; color: selectedCat === modelData ? "#FFFFFF" : Theme.text }
                                MouseArea { id: acMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: selectedCat = modelData }
                            }
                        }
                    }

                    /* All packages */
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: 16
                        spacing: 4

                        Text {
                            text: selectedTab === 1 ? "Installed Packages" : (searchText !== "" ? "Search Results" : "All Packages")
                            font.pixelSize: 14; font.bold: true; font.family: Theme.fontFamily
                            color: Theme.text
                            bottomPadding: 6
                        }

                        /* Column headers */
                        RowLayout {
                            Layout.fillWidth: true; spacing: 0; Layout.bottomMargin: 4
                            Text { Layout.preferredWidth: 44; text: ""; font.pixelSize: 10; color: Theme.textDim }
                            Text { Layout.fillWidth: true; text: "Package"; font.pixelSize: 10; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                            Text { Layout.preferredWidth: 80; text: "Category"; font.pixelSize: 10; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                            Text { Layout.preferredWidth: 60; text: "Size"; font.pixelSize: 10; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                            Text { Layout.preferredWidth: 80; text: ""; font.pixelSize: 10; color: Theme.textDim }
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                        Repeater {
                            model: getDisplayApps()

                            Rectangle {
                                Layout.fillWidth: true
                                height: 48
                                radius: Theme.radiusSmall
                                color: pkgMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Behavior on color { ColorAnimation { duration: 80 } }

                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8; spacing: 0

                                    Item {
                                        width: 36; height: 36
                                        Components.CanvasIcon {
                                            anchors.centerIn: parent
                                            iconName: modelData.ico; iconColor: modelData.clr; iconSize: 18
                                        }
                                    }

                                    Column {
                                        Layout.fillWidth: true; spacing: 1
                                        Text { text: modelData.name + "  "; font.pixelSize: 12; font.bold: true; font.family: Theme.fontFamily; color: Theme.text
                                            Text { text: modelData.ver; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textDim }
                                        }
                                        Text { text: modelData.desc; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textDim; elide: Text.ElideRight; width: 280 }
                                    }

                                    Text { Layout.preferredWidth: 80; text: modelData.cat; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textMuted }
                                    Text { Layout.preferredWidth: 60; text: modelData.size; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textDim }

                                    Rectangle {
                                        Layout.preferredWidth: 72; height: 26; radius: 13
                                        color: modelData.installed ? Theme.surfaceLight : Theme.primary
                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData.installed ? "Installed" : "Install"
                                            font.pixelSize: 10; font.bold: !modelData.installed; font.family: Theme.fontFamily
                                            color: modelData.installed ? Theme.textDim : "#FFFFFF"
                                        }
                                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor }
                                    }
                                }

                                MouseArea { id: pkgMa; anchors.fill: parent; hoverEnabled: true }
                            }
                        }
                    }

                    /* Stats */
                    Rectangle {
                        Layout.fillWidth: true; Layout.margins: 16
                        height: 50; radius: Theme.radius; color: Theme.surface

                        RowLayout {
                            anchors.fill: parent; anchors.margins: 14; spacing: 20

                            Column { spacing: 1
                                Text { text: "Total Packages"; font.pixelSize: 9; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                                Text { text: (allApps.length + featuredApps.length) + ""; font.pixelSize: 14; font.bold: true; font.family: Theme.fontFamily; color: Theme.primary }
                            }
                            Column { spacing: 1
                                Text { text: "Installed"; font.pixelSize: 9; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                                Text {
                                    text: {
                                        var count = 0
                                        for (var i = 0; i < allApps.length; i++) if (allApps[i].installed) count++
                                        for (var j = 0; j < featuredApps.length; j++) if (featuredApps[j].installed) count++
                                        return count
                                    }
                                    font.pixelSize: 14; font.bold: true; font.family: Theme.fontFamily; color: Theme.success
                                }
                            }
                            Column { spacing: 1
                                Text { text: "Updates Available"; font.pixelSize: 9; font.bold: true; font.family: Theme.fontFamily; color: Theme.textDim }
                                Text { text: "0"; font.pixelSize: 14; font.bold: true; font.family: Theme.fontFamily; color: Theme.warning }
                            }
                            Item { Layout.fillWidth: true }
                            Text { text: "Repository: NeuralOS Main • Last synced: just now"; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textMuted }
                        }
                    }

                    Item { height: 16 }
                }
            }
        }
    }
}
