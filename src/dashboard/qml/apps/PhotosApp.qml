import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: photosApp
    anchors.fill: parent

    property int selectedPhoto: -1
    property int currentTab: 0
    property bool gridMode: true
    property real previewScale: 1.0

    property var tabNames: ["All", "Favorites", "Recent", "Screenshots"]

    property var photos: [
        { title: "Sunset Beach",     date: "2026-02-01", size: "3.2 MB", w: 1920, h: 1080, c1: "#FF6B35", c2: "#F7C59F", fav: true,  cat: 0 },
        { title: "Mountain Dawn",    date: "2026-01-28", size: "4.1 MB", w: 2560, h: 1440, c1: "#2D1B69", c2: "#FF6B35", fav: false, cat: 0 },
        { title: "City Lights",      date: "2026-01-25", size: "2.8 MB", w: 1920, h: 1080, c1: "#1A0533", c2: "#5B9AFF", fav: true,  cat: 0 },
        { title: "Forest Trail",     date: "2026-01-22", size: "5.0 MB", w: 3840, h: 2160, c1: "#0D4F2B", c2: "#38F9D7", fav: false, cat: 0 },
        { title: "Ocean Horizon",    date: "2026-01-18", size: "3.6 MB", w: 2560, h: 1440, c1: "#0A1628", c2: "#4FC3F7", fav: true,  cat: 0 },
        { title: "Night Sky",        date: "2026-01-15", size: "4.4 MB", w: 3840, h: 2160, c1: "#0A0A1A", c2: "#A78BFA", fav: false, cat: 0 },
        { title: "Autumn Leaves",    date: "2026-01-10", size: "2.5 MB", w: 1920, h: 1280, c1: "#BF360C", c2: "#FFD54F", fav: true,  cat: 0 },
        { title: "Snowy Peaks",      date: "2026-01-07", size: "3.9 MB", w: 2560, h: 1440, c1: "#455A64", c2: "#E0E7FF", fav: false, cat: 0 },
        { title: "Wildflower Field", date: "2026-01-04", size: "4.7 MB", w: 3840, h: 2160, c1: "#4A148C", c2: "#E040FB", fav: false, cat: 0 },
        { title: "Desert Sunset",    date: "2025-12-30", size: "3.1 MB", w: 1920, h: 1080, c1: "#BF360C", c2: "#FF8F00", fav: true,  cat: 0 },
        { title: "Rainy Window",     date: "2025-12-27", size: "2.2 MB", w: 1920, h: 1080, c1: "#37474F", c2: "#546E7A", fav: false, cat: 0 },
        { title: "App Overview",     date: "2026-02-05", size: "0.8 MB", w: 1920, h: 1080, c1: "#161620", c2: "#5B9AFF", fav: false, cat: 3 },
        { title: "Dashboard Shot",   date: "2026-02-04", size: "1.1 MB", w: 1920, h: 1080, c1: "#161620", c2: "#34D399", fav: false, cat: 3 },
        { title: "Settings Panel",   date: "2026-02-03", size: "0.6 MB", w: 1920, h: 1080, c1: "#161620", c2: "#A78BFA", fav: false, cat: 3 },
        { title: "Coral Reef",       date: "2026-02-06", size: "5.3 MB", w: 3840, h: 2160, c1: "#006064", c2: "#00BCD4", fav: true,  cat: 0 },
        { title: "Northern Lights",  date: "2025-12-20", size: "4.8 MB", w: 2560, h: 1440, c1: "#0A0A1A", c2: "#69F0AE", fav: true,  cat: 0 },
        { title: "Cherry Blossoms",  date: "2025-12-15", size: "3.3 MB", w: 1920, h: 1280, c1: "#F8BBD0", c2: "#FCE4EC", fav: false, cat: 0 }
    ]

    function filteredPhotos() {
        if (currentTab === 0) return photos
        if (currentTab === 1) return photos.filter(function(p) { return p.fav })
        if (currentTab === 2) return photos.filter(function(p) { return p.date >= "2026-01-20" })
        if (currentTab === 3) return photos.filter(function(p) { return p.cat === 3 })
        return photos
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent; spacing: 0

            /* ---- Toolbar ---- */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 44
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 14; anchors.rightMargin: 14; spacing: 6

                    /* Album tabs */
                    Repeater {
                        model: tabNames.length

                        Rectangle {
                            width: tabLbl.implicitWidth + 20; height: 28; radius: 14
                            color: index === currentTab ? Theme.primary
                                 : tabMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                id: tabLbl; anchors.centerIn: parent
                                text: tabNames[index]
                                font.pixelSize: 12; font.family: Theme.fontFamily
                                font.weight: index === currentTab ? Font.DemiBold : Font.Normal
                                color: index === currentTab ? "#FFFFFF" : Theme.textDim
                            }

                            MouseArea {
                                id: tabMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: currentTab = index
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    /* Search placeholder */
                    Rectangle {
                        width: 140; height: 28; radius: 14
                        color: Theme.surfaceAlt

                        RowLayout {
                            anchors.centerIn: parent; spacing: 4
                            Components.CanvasIcon { iconName: "search"; iconSize: 12; iconColor: Theme.textMuted }
                            Text { text: "Search photos"; font.pixelSize: 11; font.family: Theme.fontFamily; color: Theme.textMuted }
                        }
                    }

                    /* Sort button */
                    Rectangle {
                        width: 28; height: 28; radius: Theme.radiusSmall
                        color: sortMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Components.CanvasIcon { anchors.centerIn: parent; iconName: "sort"; iconSize: 14; iconColor: Theme.textDim }
                        MouseArea { id: sortMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor }
                    }

                    /* View mode toggle */
                    Rectangle {
                        width: 28; height: 28; radius: Theme.radiusSmall
                        color: viewMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        Components.CanvasIcon { anchors.centerIn: parent; iconName: gridMode ? "list-view" : "grid-view"; iconSize: 14; iconColor: Theme.textDim }
                        MouseArea { id: viewMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: gridMode = !gridMode }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* ---- Gallery Count ---- */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 30
                color: "transparent"

                Text {
                    anchors.left: parent.left; anchors.leftMargin: 14
                    anchors.verticalCenter: parent.verticalCenter
                    text: filteredPhotos().length + " photos"
                    font.pixelSize: 11; font.family: Theme.fontFamily; color: Theme.textMuted
                }
            }

            /* ---- Grid / List Gallery ---- */
            Flickable {
                Layout.fillWidth: true; Layout.fillHeight: true
                contentHeight: gridMode ? galleryGrid.height : galleryList.height
                clip: true; flickableDirection: Flickable.VerticalFlick

                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AsNeeded; width: 5
                    contentItem: Rectangle { implicitWidth: 5; radius: 3; color: Theme.textMuted; opacity: 0.4 }
                }

                /* Grid View */
                GridLayout {
                    id: galleryGrid
                    visible: gridMode
                    width: parent.width - 24
                    x: 12
                    columns: Math.max(1, Math.floor((photosApp.width - 24) / 145))
                    rowSpacing: 8; columnSpacing: 8

                    Repeater {
                        model: filteredPhotos().length

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 140
                            radius: Theme.radiusSmall; clip: true
                            color: Theme.surfaceAlt

                            /* Gradient thumbnail */
                            Rectangle {
                                anchors.fill: parent
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: filteredPhotos()[index].c1 }
                                    GradientStop { position: 1.0; color: filteredPhotos()[index].c2 }
                                }
                            }

                            /* Title overlay */
                            Rectangle {
                                anchors.left: parent.left; anchors.right: parent.right
                                anchors.bottom: parent.bottom; height: 32
                                color: Qt.rgba(0, 0, 0, 0.55)

                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8

                                    Text {
                                        Layout.fillWidth: true
                                        text: filteredPhotos()[index].title
                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                        color: "#FFFFFF"; elide: Text.ElideRight
                                    }

                                    Components.CanvasIcon {
                                        visible: filteredPhotos()[index].fav
                                        iconName: "heart"; iconSize: 10; iconColor: "#F87171"
                                    }
                                }
                            }

                            /* Hover border */
                            Rectangle {
                                anchors.fill: parent; radius: Theme.radiusSmall
                                color: "transparent"
                                border.width: gridItemMa.containsMouse ? 2 : 0
                                border.color: Theme.primary
                            }

                            MouseArea {
                                id: gridItemMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: { selectedPhoto = index; previewScale = 1.0 }
                            }
                        }
                    }
                }

                /* List View */
                Column {
                    id: galleryList
                    visible: !gridMode
                    width: parent.width - 24; x: 12; spacing: 4

                    Repeater {
                        model: filteredPhotos().length

                        Rectangle {
                            width: parent.width; height: 52; radius: Theme.radiusSmall
                            color: listMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            RowLayout {
                                anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8; spacing: 10

                                /* Thumbnail */
                                Rectangle {
                                    width: 40; height: 36; radius: 4
                                    gradient: Gradient {
                                        orientation: Gradient.Horizontal
                                        GradientStop { position: 0.0; color: filteredPhotos()[index].c1 }
                                        GradientStop { position: 1.0; color: filteredPhotos()[index].c2 }
                                    }
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true; spacing: 2
                                    Text {
                                        text: filteredPhotos()[index].title
                                        font.pixelSize: 13; font.family: Theme.fontFamily; color: Theme.text
                                        elide: Text.ElideRight; Layout.fillWidth: true
                                    }
                                    Text {
                                        text: filteredPhotos()[index].date + "  |  " + filteredPhotos()[index].size
                                        font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textMuted
                                    }
                                }

                                Components.CanvasIcon {
                                    visible: filteredPhotos()[index].fav
                                    iconName: "heart"; iconSize: 12; iconColor: "#F87171"
                                }

                                Text {
                                    text: filteredPhotos()[index].w + " x " + filteredPhotos()[index].h
                                    font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textDim
                                }
                            }

                            MouseArea {
                                id: listMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: { selectedPhoto = index; previewScale = 1.0 }
                            }
                        }
                    }
                }
            }
        }

        /* ======== Photo Viewer Overlay ======== */
        Rectangle {
            id: viewerOverlay
            anchors.fill: parent; visible: selectedPhoto >= 0
            color: Qt.rgba(0, 0, 0, 0.85)

            MouseArea { anchors.fill: parent; onClicked: {} /* block clicks through */ }

            RowLayout {
                anchors.fill: parent; anchors.margins: 16; spacing: 12

                /* Preview area */
                Item {
                    Layout.fillWidth: true; Layout.fillHeight: true

                    Rectangle {
                        id: previewRect
                        anchors.centerIn: parent
                        width: Math.min(parent.width - 40, 500) * previewScale
                        height: Math.min(parent.height - 40, 340) * previewScale
                        radius: Theme.radiusSmall; clip: true

                        Behavior on width  { NumberAnimation { duration: Theme.animFast } }
                        Behavior on height { NumberAnimation { duration: Theme.animFast } }

                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: selectedPhoto >= 0 ? filteredPhotos()[selectedPhoto].c1 : "#000" }
                            GradientStop { position: 1.0; color: selectedPhoto >= 0 ? filteredPhotos()[selectedPhoto].c2 : "#000" }
                        }
                    }

                    /* Zoom controls */
                    Row {
                        anchors.bottom: parent.bottom; anchors.horizontalCenter: parent.horizontalCenter
                        anchors.bottomMargin: 8; spacing: 6

                        Rectangle {
                            width: 32; height: 32; radius: 16; color: Qt.rgba(0, 0, 0, 0.6)
                            Components.CanvasIcon { anchors.centerIn: parent; iconName: "zoom-out"; iconSize: 14; iconColor: "#FFFFFF" }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: previewScale = Math.max(0.4, previewScale - 0.15)
                            }
                        }

                        Rectangle {
                            width: 50; height: 32; radius: 16; color: Qt.rgba(0, 0, 0, 0.6)
                            Text {
                                anchors.centerIn: parent
                                text: Math.round(previewScale * 100) + "%"
                                font.pixelSize: 11; font.family: Theme.fontFamily; color: "#FFFFFF"
                            }
                        }

                        Rectangle {
                            width: 32; height: 32; radius: 16; color: Qt.rgba(0, 0, 0, 0.6)
                            Components.CanvasIcon { anchors.centerIn: parent; iconName: "zoom-in"; iconSize: 14; iconColor: "#FFFFFF" }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: previewScale = Math.min(2.5, previewScale + 0.15)
                            }
                        }
                    }
                }

                /* Info panel */
                Rectangle {
                    Layout.preferredWidth: 210; Layout.fillHeight: true
                    color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.95)
                    radius: Theme.radiusSmall

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 14; spacing: 10

                        /* Close button */
                        Rectangle {
                            Layout.alignment: Qt.AlignRight
                            width: 28; height: 28; radius: 14
                            color: closeMa.containsMouse ? Theme.error : Theme.surfaceAlt
                            Components.CanvasIcon { anchors.centerIn: parent; iconName: "close"; iconSize: 12; iconColor: closeMa.containsMouse ? "#FFFFFF" : Theme.textDim }
                            MouseArea { id: closeMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: selectedPhoto = -1 }
                        }

                        /* Mini preview */
                        Rectangle {
                            Layout.fillWidth: true; Layout.preferredHeight: 100
                            radius: Theme.radiusSmall; clip: true
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: selectedPhoto >= 0 ? filteredPhotos()[selectedPhoto].c1 : "#000" }
                                GradientStop { position: 1.0; color: selectedPhoto >= 0 ? filteredPhotos()[selectedPhoto].c2 : "#000" }
                            }
                        }

                        /* Title */
                        Text {
                            text: selectedPhoto >= 0 ? filteredPhotos()[selectedPhoto].title : ""
                            font.pixelSize: 16; font.weight: Font.DemiBold
                            font.family: Theme.fontFamily; color: Theme.text
                        }

                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                        /* Metadata rows */
                        Repeater {
                            model: selectedPhoto >= 0 ? [
                                { label: "Filename",   value: filteredPhotos()[selectedPhoto].title.replace(/ /g, "_") + ".jpg" },
                                { label: "Date",       value: filteredPhotos()[selectedPhoto].date },
                                { label: "Size",       value: filteredPhotos()[selectedPhoto].size },
                                { label: "Dimensions", value: filteredPhotos()[selectedPhoto].w + " x " + filteredPhotos()[selectedPhoto].h }
                            ] : []

                            ColumnLayout {
                                spacing: 2
                                Text { text: modelData.label; font.pixelSize: 10; font.family: Theme.fontFamily; color: Theme.textMuted }
                                Text { text: modelData.value; font.pixelSize: 13; font.family: Theme.fontFamily; color: Theme.text }
                            }
                        }

                        Item { Layout.fillHeight: true }

                        /* Fav badge */
                        Rectangle {
                            visible: selectedPhoto >= 0 && filteredPhotos()[selectedPhoto].fav
                            Layout.alignment: Qt.AlignHCenter
                            width: favRow.width + 16; height: 26; radius: 13
                            color: Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.15)

                            Row {
                                id: favRow; anchors.centerIn: parent; spacing: 4
                                Components.CanvasIcon { iconName: "heart"; iconSize: 12; iconColor: Theme.error }
                                Text { text: "Favorite"; font.pixelSize: 11; font.family: Theme.fontFamily; color: Theme.error }
                            }
                        }
                    }
                }
            }
        }
    }
}
