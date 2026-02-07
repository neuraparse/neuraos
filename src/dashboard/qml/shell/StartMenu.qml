import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: startMenu
    width: Theme.startMenuW
    height: Theme.startMenuH
    radius: Theme.radius
    color: Theme.startMenuBg
    border.width: 1
    border.color: Theme.glassBorder

    property var appList: []
    property var pinnedApps: []
    property var recentApps: []
    property string searchText: ""
    property string selectedCategory: ""
    property bool powerMenuOpen: false

    signal appClicked(var appDef)
    signal closeMenu()

    /* ─── Top highlight (glassmorphic inner glow) ─── */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 1
        anchors.leftMargin: 1
        anchors.rightMargin: 1
        height: 1
        radius: Theme.radius
        color: Qt.rgba(1, 1, 1, 0.05)
    }

    /* ─── Filter helpers ─── */
    function filteredApps() {
        var result = []
        for (var i = 0; i < appList.length; i++) {
            var app = appList[i]
            if (searchText !== "" && app.title.toLowerCase().indexOf(searchText.toLowerCase()) === -1)
                continue
            if (selectedCategory !== "" && app.category !== selectedCategory)
                continue
            result.push(app)
        }
        return result
    }

    function getCategories() {
        var cats = [""]
        var seen = {}
        for (var i = 0; i < appList.length; i++) {
            var c = appList[i].category
            if (!seen[c]) { seen[c] = true; cats.push(c) }
        }
        return cats
    }

    /* ─── Main layout ─── */
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 10

        /* ── 1. Header ── */
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "NeuralOS"
                font.pixelSize: Theme.fontSizeLarge
                font.bold: true
                font.family: Theme.fontFamily
                color: Theme.text
            }

            Item { Layout.fillWidth: true }

            Text {
                text: "v4.0.0"
                font.pixelSize: 10
                font.family: Theme.fontFamily
                color: Theme.textMuted
            }
        }

        /* ── 2. Search Bar ── */
        Components.SearchBar {
            Layout.fillWidth: true
            placeholder: "Search apps and files..."
            onTextChanged: startMenu.searchText = text
        }

        /* ── 3. Pinned Apps ── */
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            visible: searchText === "" && selectedCategory === "" && pinnedApps.length > 0

            Text {
                text: "Pinned"
                font.pixelSize: 11
                font.weight: Font.DemiBold
                font.family: Theme.fontFamily
                color: Theme.textDim
            }

            GridLayout {
                Layout.fillWidth: true
                columns: 6
                columnSpacing: 6
                rowSpacing: 6

                Repeater {
                    model: pinnedApps.length

                    Rectangle {
                        width: 80; height: 72
                        radius: Theme.radiusTiny
                        color: pinnedMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Column {
                            anchors.centerIn: parent
                            spacing: 4

                            Rectangle {
                                width: 42; height: 42; radius: 10
                                anchors.horizontalCenter: parent.horizontalCenter
                                color: {
                                    var c = pinnedApps[index].color || "#5B9AFF"
                                    try { var cc = Qt.color(c); return Qt.rgba(cc.r, cc.g, cc.b, Theme.darkMode ? 0.15 : 0.10) }
                                    catch(e) { return Qt.rgba(0.36, 0.60, 1.0, Theme.darkMode ? 0.15 : 0.10) }
                                }

                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: pinnedApps[index].icon
                                    iconColor: pinnedApps[index].color || Theme.primary
                                    iconSize: 20
                                }
                            }

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: pinnedApps[index].title.split(" ")[0]
                                font.pixelSize: 10
                                font.family: Theme.fontFamily
                                color: Theme.textDim
                                elide: Text.ElideRight
                                width: 72
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }

                        MouseArea {
                            id: pinnedMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: startMenu.appClicked(pinnedApps[index])
                        }
                    }
                }
            }
        }

        /* ── 4. Recent Apps ── */
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: searchText === "" && selectedCategory === "" && recentApps.length > 0

            Text {
                text: "Recent"
                font.pixelSize: 11
                font.weight: Font.DemiBold
                font.family: Theme.fontFamily
                color: Theme.textDim
            }

            Repeater {
                model: Math.min(recentApps.length, 3)

                Rectangle {
                    Layout.fillWidth: true
                    height: 38
                    radius: Theme.radiusTiny
                    color: recentMa.containsMouse ? Theme.surfaceAlt : "transparent"
                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 10
                        anchors.rightMargin: 10
                        spacing: 10

                        Components.CanvasIcon {
                            iconName: recentApps[index].icon
                            iconColor: recentApps[index].color || Theme.primary
                            iconSize: 16
                        }

                        Text {
                            Layout.fillWidth: true
                            text: recentApps[index].title
                            font.pixelSize: 13
                            font.family: Theme.fontFamily
                            color: Theme.text
                            elide: Text.ElideRight
                        }

                        Text {
                            text: recentApps[index].category
                            font.pixelSize: 10
                            font.family: Theme.fontFamily
                            color: Theme.textMuted
                        }
                    }

                    MouseArea {
                        id: recentMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: startMenu.appClicked(recentApps[index])
                    }
                }
            }
        }

        /* ── 5. Separator ── */
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: Theme.surfaceLight
        }

        /* ── 6. Categories ── */
        Row {
            Layout.fillWidth: true
            spacing: 6

            Repeater {
                model: getCategories()

                delegate: Rectangle {
                    width: catLabel.implicitWidth + 18
                    height: 28
                    radius: 14
                    color: selectedCategory === modelData
                        ? Theme.primary
                        : (catMa.containsMouse ? Theme.surfaceAlt : "transparent")
                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Text {
                        id: catLabel
                        anchors.centerIn: parent
                        text: modelData === "" ? "All" : modelData
                        font.pixelSize: 11
                        font.family: Theme.fontFamily
                        font.weight: Font.DemiBold
                        color: selectedCategory === modelData ? "#FFFFFF" : Theme.text
                    }

                    MouseArea {
                        id: catMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: selectedCategory = modelData
                    }
                }
            }
        }

        /* ── 7. App Grid (scrollable) ── */
        Flickable {
            id: appFlickable
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: appGrid.height
            clip: true
            flickableDirection: Flickable.VerticalFlick
            boundsBehavior: Flickable.StopAtBounds

            ScrollBar.vertical: ScrollBar {
                policy: appFlickable.contentHeight > appFlickable.height
                    ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                width: 4
                contentItem: Rectangle {
                    implicitWidth: 4
                    radius: 2
                    color: Theme.textMuted
                    opacity: 0.5
                }
                background: Rectangle { color: "transparent" }
            }

            GridLayout {
                id: appGrid
                width: parent.width
                columns: 3
                rowSpacing: 4
                columnSpacing: 4

                Repeater {
                    model: filteredApps()

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 60
                        radius: Theme.radiusTiny
                        color: appItemMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 10

                            Rectangle {
                                width: 40; height: 40; radius: 10
                                color: {
                                    var c = modelData.color || "#5B9AFF"
                                    try { var cc = Qt.color(c); return Qt.rgba(cc.r, cc.g, cc.b, Theme.darkMode ? 0.15 : 0.10) }
                                    catch(e) { return Qt.rgba(0.36, 0.60, 1.0, Theme.darkMode ? 0.15 : 0.10) }
                                }

                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: modelData.icon
                                    iconColor: modelData.color || Theme.primary
                                    iconSize: 20
                                }
                            }

                            Column {
                                Layout.fillWidth: true
                                spacing: 2

                                Text {
                                    text: modelData.title
                                    font.pixelSize: 13
                                    font.bold: true
                                    font.family: Theme.fontFamily
                                    color: Theme.text
                                    elide: Text.ElideRight
                                    width: parent.width
                                }

                                Text {
                                    text: modelData.category
                                    font.pixelSize: 10
                                    font.family: Theme.fontFamily
                                    color: Theme.textDim
                                }
                            }
                        }

                        MouseArea {
                            id: appItemMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: startMenu.appClicked(modelData)
                        }
                    }
                }
            }

            /* Empty state */
            Text {
                visible: filteredApps().length === 0
                anchors.centerIn: parent
                text: searchText !== ""
                    ? "No results for \"" + searchText + "\""
                    : "No applications found"
                font.pixelSize: 13
                font.family: Theme.fontFamily
                color: Theme.textMuted
            }
        }

        /* ── 8. Bottom Bar ── */
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            /* User info */
            RowLayout {
                spacing: 8

                Rectangle {
                    width: 30; height: 30; radius: 15
                    color: Theme.surfaceAlt

                    Components.CanvasIcon {
                        anchors.centerIn: parent
                        iconName: "user"
                        iconColor: Theme.textDim
                        iconSize: 15
                    }
                }

                Column {
                    spacing: 0

                    Text {
                        text: "root"
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                        font.family: Theme.fontFamily
                        color: Theme.text
                    }

                    Text {
                        text: "Administrator"
                        font.pixelSize: 9
                        font.family: Theme.fontFamily
                        color: Theme.textMuted
                    }
                }
            }

            Item { Layout.fillWidth: true }

            /* Power button with dropdown */
            Item {
                width: 38; height: 34

                Rectangle {
                    anchors.fill: parent
                    radius: Theme.radiusTiny
                    color: pwrMa.containsMouse ? Theme.error : Theme.surfaceAlt
                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                    Components.CanvasIcon {
                        anchors.centerIn: parent
                        iconName: "power"
                        iconColor: pwrMa.containsMouse ? "#FFFFFF" : Theme.textDim
                        iconSize: 15
                    }

                    MouseArea {
                        id: pwrMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: powerMenuOpen = !powerMenuOpen
                    }
                }

                /* Power dropdown menu */
                Rectangle {
                    visible: powerMenuOpen
                    anchors.bottom: parent.top
                    anchors.right: parent.right
                    anchors.bottomMargin: 8
                    width: 170
                    height: powerCol.height + 16
                    radius: Theme.radiusSmall
                    color: Theme.surface
                    border.width: 1
                    border.color: Theme.glassBorder

                    /* Top highlight inside power dropdown */
                    Rectangle {
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.topMargin: 1
                        anchors.leftMargin: 1
                        anchors.rightMargin: 1
                        height: 1
                        radius: Theme.radiusSmall
                        color: Qt.rgba(1, 1, 1, 0.05)
                    }

                    Column {
                        id: powerCol
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 8
                        spacing: 2

                        Repeater {
                            model: [
                                { label: "Lock Screen", icon: "lock",    act: "lock" },
                                { label: "Sleep",       icon: "moon",    act: "sleep" },
                                { label: "Restart",     icon: "refresh", act: "restart" },
                                { label: "Shut Down",   icon: "power",   act: "shutdown", clr: "#E84855" }
                            ]

                            Rectangle {
                                width: parent.width
                                height: 32
                                radius: Theme.radiusTiny
                                color: pmItemMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Behavior on color { ColorAnimation { duration: Theme.animFast } }

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    anchors.rightMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        iconName: modelData.icon
                                        iconSize: 14
                                        iconColor: modelData.clr || Theme.textDim
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.label
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: modelData.clr || Theme.text
                                    }
                                }

                                MouseArea {
                                    id: pmItemMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        powerMenuOpen = false
                                        if (modelData.act === "shutdown") Qt.quit()
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
