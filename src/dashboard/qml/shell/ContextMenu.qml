import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: ctxMenu
    anchors.fill: parent
    visible: menuRect.visible

    signal itemClicked(string action)

    property var menuItems: []
    property real menuX: 0
    property real menuY: 0

    function show(x, y, items) {
        menuItems = items
        menuX = Math.min(x, parent.width - 220)
        menuY = Math.min(y, parent.height - menuCol.height - 16)
        menuRect.visible = true
        openAnim.start()
    }

    function hide() {
        closeAnim.start()
    }

    /* Backdrop: click to dismiss */
    MouseArea {
        anchors.fill: parent
        visible: menuRect.visible
        onClicked: hide()
    }

    /* ─── Glass menu panel ─── */
    Rectangle {
        id: menuRect
        visible: false
        x: menuX; y: menuY
        width: 210
        height: menuCol.height + 12
        radius: 12
        color: Theme.glass
        border.width: 1
        border.color: Theme.glassBorder
        scale: 0.95; opacity: 0
        transformOrigin: Item.TopLeft

        /* Top highlight */
        Rectangle {
            anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
            anchors.leftMargin: 12; anchors.rightMargin: 12
            height: 1; color: Qt.rgba(1, 1, 1, Theme.darkMode ? 0.05 : 0.2)
        }

        ParallelAnimation {
            id: openAnim
            NumberAnimation { target: menuRect; property: "scale"; to: 1.0; duration: 120; easing.type: Easing.OutQuint }
            NumberAnimation { target: menuRect; property: "opacity"; to: 1.0; duration: 100; easing.type: Easing.OutQuint }
        }

        ParallelAnimation {
            id: closeAnim
            NumberAnimation { target: menuRect; property: "scale"; to: 0.95; duration: 80; easing.type: Easing.InQuad }
            NumberAnimation { target: menuRect; property: "opacity"; to: 0; duration: 80; easing.type: Easing.InQuad }
            onStopped: menuRect.visible = false
        }

        Column {
            id: menuCol
            anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
            anchors.margins: 6
            spacing: 2

            Repeater {
                model: menuItems

                Loader {
                    width: parent.width
                    sourceComponent: modelData.separator ? separatorComp : menuItemComp
                    property var itemData: modelData
                }
            }
        }
    }

    Component {
        id: separatorComp
        Rectangle {
            height: 9; color: "transparent"
            Rectangle {
                anchors.centerIn: parent; width: parent.width - 12; height: 1
                color: Theme.surfaceLight
            }
        }
    }

    Component {
        id: menuItemComp
        Rectangle {
            height: 32; radius: Theme.radiusTiny
            color: miMa.containsMouse ? Theme.glassHover : "transparent"
            Behavior on color { ColorAnimation { duration: 60 } }

            RowLayout {
                anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 10

                Components.CanvasIcon {
                    iconName: itemData.icon || "info"
                    iconSize: 14
                    iconColor: itemData.color || Theme.textDim
                }

                Text {
                    Layout.fillWidth: true
                    text: itemData.label || ""
                    font.pixelSize: 12; font.family: Theme.fontFamily
                    color: itemData.color || Theme.text
                    elide: Text.ElideRight
                }
            }

            MouseArea {
                id: miMa; anchors.fill: parent
                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                onClicked: { ctxMenu.itemClicked(itemData.action || ""); ctxMenu.hide() }
            }
        }
    }
}
