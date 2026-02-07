import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: fmApp
    anchors.fill: parent

    /* ── State Properties ── */
    property string currentPath: "/home/user"
    property var pathHistory: ["/home/user"]
    property int historyIndex: 0
    property bool showGrid: true
    property bool showPreview: false
    property string sortBy: "name"
    property int selectedIndex: -1
    property string searchText: ""

    /* ── Mock Filesystem ── */
    property var fileSystem: {
        "/home/user": [
            { name: "Documents", type: "folder", size: 0, date: "2026-02-01", icon: "folder", color: "#5B9AFF" },
            { name: "Downloads", type: "folder", size: 0, date: "2026-02-05", icon: "download", color: "#38BDF8" },
            { name: "Pictures", type: "folder", size: 0, date: "2026-01-20", icon: "image", color: "#F472B6" },
            { name: "Music", type: "folder", size: 0, date: "2026-01-15", icon: "volume", color: "#A78BFA" },
            { name: "Projects", type: "folder", size: 0, date: "2026-02-06", icon: "code", color: "#34D399" },
            { name: "Videos", type: "folder", size: 0, date: "2026-01-10", icon: "video", color: "#FBBF24" },
            { name: "readme.md", type: "file", size: 2048, date: "2026-02-06", icon: "file", color: "#8B8FA2" },
            { name: "config.json", type: "file", size: 512, date: "2026-02-04", icon: "code", color: "#34D399" },
            { name: "photo.png", type: "file", size: 1048576, date: "2026-02-03", icon: "image", color: "#F472B6" },
            { name: "notes.txt", type: "file", size: 1024, date: "2026-02-05", icon: "edit", color: "#8B8FA2" }
        ],
        "/home/user/Documents": [
            { name: "report.pdf", type: "file", size: 524288, date: "2026-02-01", icon: "file", color: "#F87171" },
            { name: "presentation.pptx", type: "file", size: 2097152, date: "2026-01-28", icon: "file", color: "#FBBF24" },
            { name: "budget.xlsx", type: "file", size: 102400, date: "2026-01-30", icon: "grid", color: "#34D399" },
            { name: "notes", type: "folder", size: 0, date: "2026-02-02", icon: "folder", color: "#5B9AFF" }
        ],
        "/home/user/Downloads": [
            { name: "setup.deb", type: "file", size: 52428800, date: "2026-02-05", icon: "download", color: "#38BDF8" },
            { name: "archive.tar.gz", type: "file", size: 10485760, date: "2026-02-04", icon: "box", color: "#A78BFA" },
            { name: "image.iso", type: "file", size: 734003200, date: "2026-02-03", icon: "database", color: "#F87171" }
        ],
        "/home/user/Pictures": [
            { name: "vacation", type: "folder", size: 0, date: "2026-01-20", icon: "folder", color: "#5B9AFF" },
            { name: "screenshots", type: "folder", size: 0, date: "2026-02-06", icon: "folder", color: "#5B9AFF" },
            { name: "wallpaper.jpg", type: "file", size: 2097152, date: "2026-01-18", icon: "image", color: "#F472B6" },
            { name: "avatar.png", type: "file", size: 51200, date: "2026-01-22", icon: "image", color: "#F472B6" }
        ],
        "/home/user/Projects": [
            { name: "neuraos", type: "folder", size: 0, date: "2026-02-06", icon: "code", color: "#34D399" },
            { name: "website", type: "folder", size: 0, date: "2026-01-25", icon: "globe", color: "#38BDF8" },
            { name: "ml-model", type: "folder", size: 0, date: "2026-02-01", icon: "neural", color: "#A78BFA" }
        ]
    }

    /* ══════════════════════════════════════════════════════
       FUNCTIONS
       ══════════════════════════════════════════════════════ */

    function navigateTo(path) {
        if (currentPath === path) return
        /* Trim forward history when navigating from mid-history */
        var newHistory = pathHistory.slice(0, historyIndex + 1)
        newHistory.push(path)
        pathHistory = newHistory
        historyIndex = pathHistory.length - 1
        currentPath = path
        selectedIndex = -1
        searchText = ""
    }

    function goBack() {
        if (historyIndex > 0) {
            historyIndex--
            currentPath = pathHistory[historyIndex]
            selectedIndex = -1
        }
    }

    function goForward() {
        if (historyIndex < pathHistory.length - 1) {
            historyIndex++
            currentPath = pathHistory[historyIndex]
            selectedIndex = -1
        }
    }

    function goUp() {
        if (currentPath === "/") return
        var parts = currentPath.split("/")
        parts.pop()
        var parent = parts.join("/")
        if (parent === "") parent = "/"
        navigateTo(parent)
    }

    function formatSize(bytes) {
        if (bytes <= 0) return "--"
        if (bytes < 1024) return bytes + " B"
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB"
        return (bytes / 1073741824).toFixed(1) + " GB"
    }

    function getCurrentFiles() {
        var files = fileSystem[currentPath]
        if (!files) return []

        /* Filter by search text */
        if (searchText.length > 0) {
            var query = searchText.toLowerCase()
            files = files.filter(function(f) {
                return f.name.toLowerCase().indexOf(query) >= 0
            })
        }

        /* Sort */
        var sorted = files.slice()
        sorted.sort(function(a, b) {
            /* Folders always first */
            if (a.type === "folder" && b.type !== "folder") return -1
            if (a.type !== "folder" && b.type === "folder") return 1

            if (sortBy === "name") {
                return a.name.toLowerCase().localeCompare(b.name.toLowerCase())
            } else if (sortBy === "size") {
                return a.size - b.size
            } else if (sortBy === "date") {
                return a.date.localeCompare(b.date)
            } else if (sortBy === "type") {
                var extA = a.name.indexOf(".") >= 0 ? a.name.split(".").pop() : ""
                var extB = b.name.indexOf(".") >= 0 ? b.name.split(".").pop() : ""
                return extA.localeCompare(extB)
            }
            return 0
        })

        return sorted
    }

    function getPathSegments() {
        if (currentPath === "/") return ["/"]
        var parts = currentPath.split("/").filter(function(s) { return s.length > 0 })
        return ["/"].concat(parts)
    }

    function pathForSegment(index) {
        var segments = getPathSegments()
        if (index === 0) return "/"
        var parts = segments.slice(1, index + 1)
        return "/" + parts.join("/")
    }

    /* ══════════════════════════════════════════════════════
       MAIN LAYOUT
       ══════════════════════════════════════════════════════ */

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ═══════════════════════════════════════════════
               TOOLBAR
               ═══════════════════════════════════════════════ */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10
                    anchors.rightMargin: 10
                    spacing: 4

                    /* Back Button */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: backBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        opacity: historyIndex > 0 ? 1.0 : 0.35

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "arrow-left"
                            iconColor: Theme.text
                            iconSize: 16
                        }

                        MouseArea {
                            id: backBtnMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: historyIndex > 0 ? Qt.PointingHandCursor : Qt.ArrowCursor
                            enabled: historyIndex > 0
                            onClicked: goBack()
                        }
                    }

                    /* Forward Button */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: fwdBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        opacity: historyIndex < pathHistory.length - 1 ? 1.0 : 0.35

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "arrow-right"
                            iconColor: Theme.text
                            iconSize: 16
                        }

                        MouseArea {
                            id: fwdBtnMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: historyIndex < pathHistory.length - 1 ? Qt.PointingHandCursor : Qt.ArrowCursor
                            enabled: historyIndex < pathHistory.length - 1
                            onClicked: goForward()
                        }
                    }

                    /* Up Button */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: upBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        opacity: currentPath !== "/" ? 1.0 : 0.35

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "arrow-up"
                            iconColor: Theme.text
                            iconSize: 16
                        }

                        MouseArea {
                            id: upBtnMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: currentPath !== "/" ? Qt.PointingHandCursor : Qt.ArrowCursor
                            enabled: currentPath !== "/"
                            onClicked: goUp()
                        }
                    }

                    /* Separator */
                    Rectangle {
                        width: 1; height: 22
                        color: Theme.surfaceLight
                        Layout.leftMargin: 4
                        Layout.rightMargin: 4
                    }

                    /* Breadcrumb */
                    Flickable {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 32
                        contentWidth: breadcrumbRow.width
                        clip: true
                        flickableDirection: Flickable.HorizontalFlick
                        boundsBehavior: Flickable.StopAtBounds

                        Row {
                            id: breadcrumbRow
                            height: 32
                            spacing: 2

                            Repeater {
                                model: getPathSegments().length

                                Row {
                                    height: 32
                                    spacing: 2

                                    /* Separator chevron (skip for first) */
                                    Text {
                                        visible: index > 0
                                        text: "\u203A"
                                        color: Theme.textMuted
                                        font.pixelSize: 16
                                        font.family: Theme.fontFamily
                                        anchors.verticalCenter: parent.verticalCenter
                                    }

                                    /* Segment button */
                                    Rectangle {
                                        height: 26
                                        width: segLabel.implicitWidth + 16
                                        radius: Theme.radiusTiny
                                        anchors.verticalCenter: parent.verticalCenter
                                        color: segMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                        Behavior on color { ColorAnimation { duration: 80 } }

                                        Text {
                                            id: segLabel
                                            anchors.centerIn: parent
                                            text: {
                                                var segs = getPathSegments()
                                                if (index === 0) return "/"
                                                return segs[index] || ""
                                            }
                                            font.pixelSize: 12
                                            font.family: Theme.fontFamily
                                            font.bold: index === getPathSegments().length - 1
                                            color: index === getPathSegments().length - 1 ? Theme.text : Theme.textDim
                                        }

                                        MouseArea {
                                            id: segMa
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: {
                                                var target = pathForSegment(index)
                                                navigateTo(target)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* Separator */
                    Rectangle {
                        width: 1; height: 22
                        color: Theme.surfaceLight
                        Layout.leftMargin: 4
                        Layout.rightMargin: 4
                    }

                    /* Search Field */
                    Rectangle {
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 30
                        radius: Theme.radiusSmall
                        color: Theme.surfaceAlt
                        border.width: searchInput.activeFocus ? 1 : 0
                        border.color: Theme.primary

                        Behavior on border.width { NumberAnimation { duration: 80 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 8
                            anchors.rightMargin: 8
                            spacing: 6

                            Components.CanvasIcon {
                                iconName: "search"
                                iconSize: 13
                                iconColor: Theme.textMuted
                            }

                            TextInput {
                                id: searchInput
                                Layout.fillWidth: true
                                Layout.alignment: Qt.AlignVCenter
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                                color: Theme.text
                                clip: true
                                selectByMouse: true
                                selectedTextColor: "#FFFFFF"
                                selectionColor: Theme.primary
                                onTextChanged: fmApp.searchText = text

                                Text {
                                    anchors.fill: parent
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "Search..."
                                    color: Theme.textMuted
                                    font.pixelSize: 12
                                    font.family: Theme.fontFamily
                                    visible: !searchInput.text && !searchInput.activeFocus
                                    verticalAlignment: Text.AlignVCenter
                                }
                            }

                            /* Clear button */
                            Rectangle {
                                width: 16; height: 16
                                radius: 8
                                visible: searchInput.text.length > 0
                                color: clearSearchMa.containsMouse ? Theme.surfaceLight : "transparent"

                                Text {
                                    anchors.centerIn: parent
                                    text: "\u00D7"
                                    font.pixelSize: 12
                                    color: Theme.textDim
                                }

                                MouseArea {
                                    id: clearSearchMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: { searchInput.text = ""; searchInput.focus = false }
                                }
                            }
                        }
                    }

                    /* Separator */
                    Rectangle {
                        width: 1; height: 22
                        color: Theme.surfaceLight
                        Layout.leftMargin: 2
                        Layout.rightMargin: 2
                    }

                    /* Grid View Toggle */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: showGrid
                              ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                              : gridToggleMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "grid-view"
                            iconSize: 15
                            iconColor: showGrid ? Theme.primary : Theme.textDim
                        }

                        MouseArea {
                            id: gridToggleMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: showGrid = true
                        }
                    }

                    /* List View Toggle */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: !showGrid
                              ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                              : listToggleMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "list-view"
                            iconSize: 15
                            iconColor: !showGrid ? Theme.primary : Theme.textDim
                        }

                        MouseArea {
                            id: listToggleMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: showGrid = false
                        }
                    }

                    /* Preview Toggle */
                    Rectangle {
                        width: 32; height: 32
                        radius: Theme.radiusSmall
                        color: showPreview
                              ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                              : previewToggleMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Behavior on color { ColorAnimation { duration: Theme.animFast } }

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "sidebar"
                            iconSize: 15
                            iconColor: showPreview ? Theme.primary : Theme.textDim
                        }

                        MouseArea {
                            id: previewToggleMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: showPreview = !showPreview
                        }
                    }
                }
            }

            /* Toolbar separator */
            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* ═══════════════════════════════════════════════
               MAIN CONTENT ROW: Sidebar + Content + Preview
               ═══════════════════════════════════════════════ */
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* ───────────────────────────────────────────
                   SIDEBAR (220px)
                   ─────────────────────────────────────────── */
                Rectangle {
                    Layout.fillHeight: true
                    Layout.preferredWidth: 220
                    color: Theme.surfaceAlt

                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 10
                        contentHeight: sidebarCol.height
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        Column {
                            id: sidebarCol
                            width: parent.width
                            spacing: 2

                            /* ── Favorites Section ── */
                            Text {
                                text: "FAVORITES"
                                color: Theme.textDim
                                font.pixelSize: 10
                                font.bold: true
                                font.family: Theme.fontFamily
                                font.letterSpacing: 0.5
                                topPadding: 4
                                bottomPadding: 8
                                leftPadding: 8
                            }

                            /* Home */
                            Rectangle {
                                width: parent.width
                                height: 34
                                radius: Theme.radiusTiny
                                color: currentPath === "/home/user"
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    : homeMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "home"
                                        iconSize: 15
                                        iconColor: currentPath === "/home/user" ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Home"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: currentPath === "/home/user" ? Theme.primary : Theme.text
                                        font.bold: currentPath === "/home/user"
                                    }
                                }

                                MouseArea {
                                    id: homeMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/home/user")
                                }
                            }

                            /* Documents */
                            Rectangle {
                                width: parent.width
                                height: 34
                                radius: Theme.radiusTiny
                                color: currentPath === "/home/user/Documents"
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    : docsMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "file"
                                        iconSize: 15
                                        iconColor: currentPath === "/home/user/Documents" ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Documents"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: currentPath === "/home/user/Documents" ? Theme.primary : Theme.text
                                        font.bold: currentPath === "/home/user/Documents"
                                    }
                                }

                                MouseArea {
                                    id: docsMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/home/user/Documents")
                                }
                            }

                            /* Downloads */
                            Rectangle {
                                width: parent.width
                                height: 34
                                radius: Theme.radiusTiny
                                color: currentPath === "/home/user/Downloads"
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    : dlMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "download"
                                        iconSize: 15
                                        iconColor: currentPath === "/home/user/Downloads" ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Downloads"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: currentPath === "/home/user/Downloads" ? Theme.primary : Theme.text
                                        font.bold: currentPath === "/home/user/Downloads"
                                    }
                                }

                                MouseArea {
                                    id: dlMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/home/user/Downloads")
                                }
                            }

                            /* Pictures */
                            Rectangle {
                                width: parent.width
                                height: 34
                                radius: Theme.radiusTiny
                                color: currentPath === "/home/user/Pictures"
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    : picsMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "image"
                                        iconSize: 15
                                        iconColor: currentPath === "/home/user/Pictures" ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Pictures"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: currentPath === "/home/user/Pictures" ? Theme.primary : Theme.text
                                        font.bold: currentPath === "/home/user/Pictures"
                                    }
                                }

                                MouseArea {
                                    id: picsMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/home/user/Pictures")
                                }
                            }

                            /* Music */
                            Rectangle {
                                width: parent.width
                                height: 34
                                radius: Theme.radiusTiny
                                color: currentPath === "/home/user/Music"
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    : musicMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    spacing: 10

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "volume"
                                        iconSize: 15
                                        iconColor: currentPath === "/home/user/Music" ? Theme.primary : Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Music"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: currentPath === "/home/user/Music" ? Theme.primary : Theme.text
                                        font.bold: currentPath === "/home/user/Music"
                                    }
                                }

                                MouseArea {
                                    id: musicMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/home/user/Music")
                                }
                            }

                            /* ── Separator ── */
                            Item { width: 1; height: 12 }

                            Rectangle {
                                width: parent.width - 16
                                anchors.horizontalCenter: parent.horizontalCenter
                                height: 1
                                color: Theme.surfaceLight
                            }

                            Item { width: 1; height: 12 }

                            /* ── Drives Section ── */
                            Text {
                                text: "DRIVES"
                                color: Theme.textDim
                                font.pixelSize: 10
                                font.bold: true
                                font.family: Theme.fontFamily
                                font.letterSpacing: 0.5
                                bottomPadding: 8
                                leftPadding: 8
                            }

                            /* Root drive */
                            Rectangle {
                                width: parent.width
                                height: 52
                                radius: Theme.radiusTiny
                                color: rootDriveMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Column {
                                    anchors.left: parent.left
                                    anchors.leftMargin: 10
                                    anchors.right: parent.right
                                    anchors.rightMargin: 10
                                    anchors.verticalCenter: parent.verticalCenter
                                    spacing: 4

                                    Row {
                                        spacing: 8
                                        Components.CanvasIcon {
                                            anchors.verticalCenter: parent.verticalCenter
                                            iconName: "database"
                                            iconSize: 14
                                            iconColor: Theme.textDim
                                        }
                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: "/ (root)"
                                            font.pixelSize: 12
                                            font.family: Theme.fontFamily
                                            color: Theme.text
                                        }
                                        Item { width: 1; height: 1 }
                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: "250 GB"
                                            font.pixelSize: 10
                                            font.family: Theme.fontFamily
                                            color: Theme.textDim
                                        }
                                    }

                                    /* Usage bar */
                                    Rectangle {
                                        width: parent.width
                                        height: 4
                                        radius: 2
                                        color: Theme.surfaceLight

                                        Rectangle {
                                            width: parent.width * 0.42
                                            height: parent.height
                                            radius: 2
                                            color: Theme.primary
                                        }
                                    }
                                }

                                MouseArea {
                                    id: rootDriveMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: navigateTo("/")
                                }
                            }

                            /* Data drive */
                            Rectangle {
                                width: parent.width
                                height: 52
                                radius: Theme.radiusTiny
                                color: dataDriveMa.containsMouse ? Theme.hoverBg : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Column {
                                    anchors.left: parent.left
                                    anchors.leftMargin: 10
                                    anchors.right: parent.right
                                    anchors.rightMargin: 10
                                    anchors.verticalCenter: parent.verticalCenter
                                    spacing: 4

                                    Row {
                                        spacing: 8
                                        Components.CanvasIcon {
                                            anchors.verticalCenter: parent.verticalCenter
                                            iconName: "database"
                                            iconSize: 14
                                            iconColor: Theme.textDim
                                        }
                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: "/mnt/data"
                                            font.pixelSize: 12
                                            font.family: Theme.fontFamily
                                            color: Theme.text
                                        }
                                        Item { width: 1; height: 1 }
                                        Text {
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: "1 TB"
                                            font.pixelSize: 10
                                            font.family: Theme.fontFamily
                                            color: Theme.textDim
                                        }
                                    }

                                    /* Usage bar */
                                    Rectangle {
                                        width: parent.width
                                        height: 4
                                        radius: 2
                                        color: Theme.surfaceLight

                                        Rectangle {
                                            width: parent.width * 0.28
                                            height: parent.height
                                            radius: 2
                                            color: Theme.secondary
                                        }
                                    }
                                }

                                MouseArea {
                                    id: dataDriveMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }
                    }
                }

                /* Sidebar separator */
                Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

                /* ───────────────────────────────────────────
                   CONTENT AREA
                   ─────────────────────────────────────────── */
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    /* Empty state */
                    Column {
                        anchors.centerIn: parent
                        spacing: 12
                        visible: getCurrentFiles().length === 0

                        Components.CanvasIcon {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconName: searchText.length > 0 ? "search" : "folder"
                            iconSize: 48
                            iconColor: Theme.textMuted
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: searchText.length > 0 ? "No results for \"" + searchText + "\"" : "This folder is empty"
                            font.pixelSize: 14
                            font.family: Theme.fontFamily
                            color: Theme.textDim
                        }
                    }

                    /* ── Grid View ── */
                    Flickable {
                        id: gridFlickable
                        anchors.fill: parent
                        visible: showGrid && getCurrentFiles().length > 0
                        contentHeight: gridFlow.height + 24
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        Flow {
                            id: gridFlow
                            width: parent.width
                            padding: 12
                            spacing: 6

                            Repeater {
                                model: getCurrentFiles().length

                                Rectangle {
                                    id: gridItem
                                    width: 88; height: 88
                                    radius: Theme.radiusSmall
                                    color: selectedIndex === index
                                          ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.12)
                                          : gridItemMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                    border.width: selectedIndex === index ? 1.5 : 0
                                    border.color: Theme.primary

                                    Behavior on color { ColorAnimation { duration: 80 } }

                                    property var fileData: getCurrentFiles()[index]

                                    Column {
                                        anchors.centerIn: parent
                                        spacing: 6

                                        /* Icon circle */
                                        Rectangle {
                                            width: 36; height: 36
                                            radius: 18
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            color: fileData ? Qt.rgba(
                                                       Qt.darker(fileData.color, 1.0).r,
                                                       Qt.darker(fileData.color, 1.0).g,
                                                       Qt.darker(fileData.color, 1.0).b,
                                                       0.18) : "transparent"

                                            Components.CanvasIcon {
                                                anchors.centerIn: parent
                                                iconName: fileData ? fileData.icon : ""
                                                iconSize: 18
                                                iconColor: fileData ? fileData.color : Theme.textDim
                                            }
                                        }

                                        /* File name */
                                        Text {
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            width: 78
                                            text: fileData ? fileData.name : ""
                                            font.pixelSize: 10
                                            font.family: Theme.fontFamily
                                            color: selectedIndex === index ? Theme.primary : Theme.text
                                            horizontalAlignment: Text.AlignHCenter
                                            elide: Text.ElideMiddle
                                            maximumLineCount: 1
                                        }
                                    }

                                    MouseArea {
                                        id: gridItemMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            selectedIndex = index
                                        }
                                        onDoubleClicked: {
                                            if (fileData && fileData.type === "folder") {
                                                var target = currentPath === "/"
                                                    ? "/" + fileData.name
                                                    : currentPath + "/" + fileData.name
                                                navigateTo(target)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* ── List View ── */
                    ColumnLayout {
                        anchors.fill: parent
                        visible: !showGrid && getCurrentFiles().length > 0
                        spacing: 0

                        /* Column Header */
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 32
                            color: Theme.surfaceAlt

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 12
                                anchors.rightMargin: 12
                                spacing: 0

                                /* Icon spacer */
                                Item { Layout.preferredWidth: 32; Layout.fillHeight: true }

                                /* Name header */
                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    color: nameHeaderMa.containsMouse ? Theme.hoverBg : "transparent"
                                    radius: Theme.radiusTiny

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 4
                                        text: "Name" + (sortBy === "name" ? "  \u2193" : "")
                                        font.pixelSize: 11
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                        color: sortBy === "name" ? Theme.primary : Theme.textDim
                                    }

                                    MouseArea {
                                        id: nameHeaderMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: sortBy = "name"
                                    }
                                }

                                /* Size header */
                                Rectangle {
                                    Layout.preferredWidth: 90
                                    Layout.fillHeight: true
                                    color: sizeHeaderMa.containsMouse ? Theme.hoverBg : "transparent"
                                    radius: Theme.radiusTiny

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 4
                                        text: "Size" + (sortBy === "size" ? "  \u2193" : "")
                                        font.pixelSize: 11
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                        color: sortBy === "size" ? Theme.primary : Theme.textDim
                                    }

                                    MouseArea {
                                        id: sizeHeaderMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: sortBy = "size"
                                    }
                                }

                                /* Date header */
                                Rectangle {
                                    Layout.preferredWidth: 100
                                    Layout.fillHeight: true
                                    color: dateHeaderMa.containsMouse ? Theme.hoverBg : "transparent"
                                    radius: Theme.radiusTiny

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 4
                                        text: "Date" + (sortBy === "date" ? "  \u2193" : "")
                                        font.pixelSize: 11
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                        color: sortBy === "date" ? Theme.primary : Theme.textDim
                                    }

                                    MouseArea {
                                        id: dateHeaderMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: sortBy = "date"
                                    }
                                }

                                /* Type header */
                                Rectangle {
                                    Layout.preferredWidth: 70
                                    Layout.fillHeight: true
                                    color: typeHeaderMa.containsMouse ? Theme.hoverBg : "transparent"
                                    radius: Theme.radiusTiny

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 4
                                        text: "Type" + (sortBy === "type" ? "  \u2193" : "")
                                        font.pixelSize: 11
                                        font.bold: true
                                        font.family: Theme.fontFamily
                                        color: sortBy === "type" ? Theme.primary : Theme.textDim
                                    }

                                    MouseArea {
                                        id: typeHeaderMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: sortBy = "type"
                                    }
                                }
                            }
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                        /* File Rows */
                        Flickable {
                            id: listFlickable
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            contentHeight: listCol.height
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds

                            Column {
                                id: listCol
                                width: parent.width

                                Repeater {
                                    model: getCurrentFiles().length

                                    Rectangle {
                                        id: listRow
                                        width: parent.width
                                        height: 34
                                        color: {
                                            if (selectedIndex === index) {
                                                return Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                                            }
                                            if (listRowMa.containsMouse) {
                                                return Theme.surfaceAlt
                                            }
                                            return index % 2 === 0 ? "transparent"
                                                                   : Qt.rgba(Theme.surfaceAlt.r, Theme.surfaceAlt.g, Theme.surfaceAlt.b, 0.4)
                                        }

                                        property var fileData: getCurrentFiles()[index]

                                        RowLayout {
                                            anchors.fill: parent
                                            anchors.leftMargin: 12
                                            anchors.rightMargin: 12
                                            spacing: 0

                                            /* Icon */
                                            Item {
                                                Layout.preferredWidth: 32
                                                Layout.fillHeight: true

                                                Components.CanvasIcon {
                                                    anchors.centerIn: parent
                                                    iconName: fileData ? fileData.icon : ""
                                                    iconSize: 16
                                                    iconColor: fileData ? fileData.color : Theme.textDim
                                                }
                                            }

                                            /* Name */
                                            Text {
                                                Layout.fillWidth: true
                                                text: fileData ? fileData.name : ""
                                                font.pixelSize: 12
                                                font.family: Theme.fontFamily
                                                font.bold: fileData && fileData.type === "folder"
                                                color: {
                                                    if (selectedIndex === index) return "#FFFFFF"
                                                    if (fileData && fileData.type === "folder") return Theme.primary
                                                    return Theme.text
                                                }
                                                elide: Text.ElideRight
                                                verticalAlignment: Text.AlignVCenter
                                                leftPadding: 4
                                            }

                                            /* Size */
                                            Text {
                                                Layout.preferredWidth: 90
                                                text: fileData ? formatSize(fileData.size) : ""
                                                font.pixelSize: 11
                                                font.family: Theme.fontFamily
                                                color: selectedIndex === index ? Qt.rgba(1,1,1,0.75) : Theme.textDim
                                                verticalAlignment: Text.AlignVCenter
                                                leftPadding: 4
                                            }

                                            /* Date */
                                            Text {
                                                Layout.preferredWidth: 100
                                                text: fileData ? fileData.date : ""
                                                font.pixelSize: 11
                                                font.family: Theme.fontFamily
                                                color: selectedIndex === index ? Qt.rgba(1,1,1,0.75) : Theme.textDim
                                                verticalAlignment: Text.AlignVCenter
                                                leftPadding: 4
                                            }

                                            /* Type */
                                            Text {
                                                Layout.preferredWidth: 70
                                                text: {
                                                    if (!fileData) return ""
                                                    if (fileData.type === "folder") return "Folder"
                                                    var ext = fileData.name.split(".").pop().toUpperCase()
                                                    return ext
                                                }
                                                font.pixelSize: 11
                                                font.family: Theme.fontFamily
                                                color: selectedIndex === index ? Qt.rgba(1,1,1,0.75) : Theme.textDim
                                                verticalAlignment: Text.AlignVCenter
                                                leftPadding: 4
                                            }
                                        }

                                        MouseArea {
                                            id: listRowMa
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: {
                                                selectedIndex = index
                                            }
                                            onDoubleClicked: {
                                                if (fileData && fileData.type === "folder") {
                                                    var target = currentPath === "/"
                                                        ? "/" + fileData.name
                                                        : currentPath + "/" + fileData.name
                                                    navigateTo(target)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /* Preview separator */
                Rectangle {
                    width: 1
                    Layout.fillHeight: true
                    color: Theme.surfaceLight
                    visible: showPreview
                }

                /* ───────────────────────────────────────────
                   PREVIEW PANEL (240px)
                   ─────────────────────────────────────────── */
                Rectangle {
                    Layout.fillHeight: true
                    Layout.preferredWidth: 240
                    visible: showPreview
                    color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 0

                        /* No selection state */
                        Column {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            spacing: 12
                            visible: selectedIndex < 0 || selectedIndex >= getCurrentFiles().length

                            Item { Layout.fillHeight: true; width: 1; height: 1 }

                            Components.CanvasIcon {
                                anchors.horizontalCenter: parent.horizontalCenter
                                iconName: "eye"
                                iconSize: 36
                                iconColor: Theme.textMuted
                            }

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "Select a file\nto preview"
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                                color: Theme.textDim
                                horizontalAlignment: Text.AlignHCenter
                                lineHeight: 1.4
                            }

                            Item { Layout.fillHeight: true; width: 1; height: 1 }
                        }

                        /* Preview content (when item selected) */
                        Column {
                            Layout.fillWidth: true
                            spacing: 14
                            visible: selectedIndex >= 0 && selectedIndex < getCurrentFiles().length

                            property var previewFile: {
                                var files = getCurrentFiles()
                                if (selectedIndex >= 0 && selectedIndex < files.length) return files[selectedIndex]
                                return null
                            }

                            Item { width: 1; height: 8 }

                            /* Large icon */
                            Rectangle {
                                width: 64; height: 64
                                radius: 32
                                anchors.horizontalCenter: parent.horizontalCenter
                                color: parent.previewFile ? Qt.rgba(
                                           Qt.darker(parent.previewFile.color, 1.0).r,
                                           Qt.darker(parent.previewFile.color, 1.0).g,
                                           Qt.darker(parent.previewFile.color, 1.0).b,
                                           0.18) : Theme.surfaceAlt

                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: parent.parent.previewFile ? parent.parent.previewFile.icon : ""
                                    iconSize: 28
                                    iconColor: parent.parent.previewFile ? parent.parent.previewFile.color : Theme.textDim
                                }
                            }

                            /* File name */
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: parent.previewFile ? parent.previewFile.name : ""
                                font.pixelSize: 14
                                font.bold: true
                                font.family: Theme.fontFamily
                                color: Theme.text
                                horizontalAlignment: Text.AlignHCenter
                                width: parent.width
                                elide: Text.ElideMiddle
                            }

                            Item { width: 1; height: 4 }

                            /* Info rows */
                            Column {
                                width: parent.width
                                spacing: 10

                                /* Type */
                                Row {
                                    width: parent.width
                                    spacing: 0

                                    Text {
                                        width: parent.width * 0.4
                                        text: "Type"
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                    }
                                    Text {
                                        width: parent.width * 0.6
                                        text: {
                                            var f = parent.parent.parent.previewFile
                                            if (!f) return ""
                                            if (f.type === "folder") return "Folder"
                                            var ext = f.name.split(".").pop().toUpperCase()
                                            return ext + " File"
                                        }
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                        elide: Text.ElideRight
                                    }
                                }

                                /* Size */
                                Row {
                                    width: parent.width
                                    spacing: 0

                                    Text {
                                        width: parent.width * 0.4
                                        text: "Size"
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                    }
                                    Text {
                                        width: parent.width * 0.6
                                        text: {
                                            var f = parent.parent.parent.previewFile
                                            if (!f) return ""
                                            return formatSize(f.size)
                                        }
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }
                                }

                                /* Modified */
                                Row {
                                    width: parent.width
                                    spacing: 0

                                    Text {
                                        width: parent.width * 0.4
                                        text: "Modified"
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                    }
                                    Text {
                                        width: parent.width * 0.6
                                        text: {
                                            var f = parent.parent.parent.previewFile
                                            if (!f) return ""
                                            return f.date
                                        }
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }
                                }

                                /* Path */
                                Row {
                                    width: parent.width
                                    spacing: 0

                                    Text {
                                        width: parent.width * 0.4
                                        text: "Path"
                                        font.pixelSize: 11
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                    }
                                    Text {
                                        width: parent.width * 0.6
                                        text: {
                                            var f = parent.parent.parent.previewFile
                                            if (!f) return ""
                                            return currentPath + "/" + f.name
                                        }
                                        font.pixelSize: 10
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        elide: Text.ElideMiddle
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 2
                                    }
                                }
                            }

                            /* Separator */
                            Item { width: 1; height: 4 }

                            Rectangle {
                                width: parent.width
                                height: 1
                                color: Theme.surfaceLight
                            }

                            Item { width: 1; height: 4 }

                            /* Quick Actions */
                            Text {
                                text: "Quick Actions"
                                font.pixelSize: 10
                                font.bold: true
                                font.family: Theme.fontFamily
                                color: Theme.textDim
                                bottomPadding: 4
                            }

                            /* Open button */
                            Rectangle {
                                width: parent.width
                                height: 32
                                radius: Theme.radiusTiny
                                color: openBtnMa.containsMouse
                                      ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                                      : Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.1)

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 6

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "folder"
                                        iconSize: 13
                                        iconColor: Theme.primary
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Open"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: Theme.primary
                                        font.bold: true
                                    }
                                }

                                MouseArea {
                                    id: openBtnMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        var f = parent.parent.previewFile
                                        if (f && f.type === "folder") {
                                            var target = currentPath === "/"
                                                ? "/" + f.name
                                                : currentPath + "/" + f.name
                                            navigateTo(target)
                                        }
                                    }
                                }
                            }

                            /* Copy Path button */
                            Rectangle {
                                width: parent.width
                                height: 32
                                radius: Theme.radiusTiny
                                color: copyPathMa.containsMouse ? Theme.surfaceAlt : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 6

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "copy"
                                        iconSize: 13
                                        iconColor: Theme.textDim
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Copy Path"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }
                                }

                                MouseArea {
                                    id: copyPathMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }

                            /* Delete button */
                            Rectangle {
                                width: parent.width
                                height: 32
                                radius: Theme.radiusTiny
                                color: delBtnMa.containsMouse
                                      ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)
                                      : "transparent"

                                Behavior on color { ColorAnimation { duration: 80 } }

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 6

                                    Components.CanvasIcon {
                                        anchors.verticalCenter: parent.verticalCenter
                                        iconName: "trash"
                                        iconSize: 13
                                        iconColor: Theme.error
                                    }
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        text: "Delete"
                                        font.pixelSize: 12
                                        font.family: Theme.fontFamily
                                        color: Theme.error
                                    }
                                }

                                MouseArea {
                                    id: delBtnMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }

                            /* Fill remaining space */
                            Item { Layout.fillHeight: true; width: 1; height: 1 }
                        }
                    }
                }
            }

            /* ═══════════════════════════════════════════════
               STATUS BAR
               ═══════════════════════════════════════════════ */
            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 28
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12
                    spacing: 16

                    Text {
                        text: {
                            var files = getCurrentFiles()
                            var folders = files.filter(function(f) { return f.type === "folder" }).length
                            var fileCount = files.length - folders
                            var parts = []
                            if (folders > 0) parts.push(folders + (folders === 1 ? " folder" : " folders"))
                            if (fileCount > 0) parts.push(fileCount + (fileCount === 1 ? " file" : " files"))
                            return parts.join(", ") || "Empty"
                        }
                        font.pixelSize: 11
                        font.family: Theme.fontFamily
                        color: Theme.textDim
                    }

                    Rectangle { width: 1; height: 14; color: Theme.surfaceLight }

                    Text {
                        visible: selectedIndex >= 0 && selectedIndex < getCurrentFiles().length
                        text: {
                            var files = getCurrentFiles()
                            if (selectedIndex >= 0 && selectedIndex < files.length) {
                                return "\"" + files[selectedIndex].name + "\" selected"
                            }
                            return ""
                        }
                        font.pixelSize: 11
                        font.family: Theme.fontFamily
                        color: Theme.primary
                    }

                    Item { Layout.fillWidth: true }

                    Text {
                        text: currentPath
                        font.pixelSize: 10
                        font.family: "monospace"
                        color: Theme.textMuted
                        elide: Text.ElideMiddle
                        Layout.maximumWidth: 300
                    }
                }
            }
        }
    }
}
