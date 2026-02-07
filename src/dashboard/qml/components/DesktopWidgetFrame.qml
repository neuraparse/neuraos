import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: widgetFrame
    width: 200; height: 160
    radius: Theme.desktopWidgetRadius
    color: Theme.desktopWidgetBg
    border.width: 1
    border.color: Theme.glassBorder

    property string widgetTitle: ""
    property bool collapsed: false
    default property alias contentData: contentArea.data

    Behavior on height { NumberAnimation { duration: Theme.animFast; easing.type: Easing.OutCubic } }

    /* Top highlight (glassmorphic) */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 1; anchors.leftMargin: 1; anchors.rightMargin: 1
        height: 1
        radius: parent.radius
        color: Qt.rgba(1, 1, 1, Theme.darkMode ? 0.05 : 0.3)
    }

    /* Title bar */
    Rectangle {
        id: wHeader
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: widgetTitle ? 28 : 0
        color: "transparent"
        visible: widgetTitle !== ""

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 10
            anchors.rightMargin: 6
            spacing: 6

            Text {
                Layout.fillWidth: true
                text: widgetTitle
                color: Theme.textDim
                font.pixelSize: 10
                font.weight: Font.DemiBold
                font.letterSpacing: 0.5
                font.family: Theme.fontFamily
            }

            /* Collapse button */
            Rectangle {
                width: 18; height: 18
                radius: 4
                color: collapseMa.containsMouse ? Theme.glassHover : "transparent"

                Components.CanvasIcon {
                    anchors.centerIn: parent
                    iconName: collapsed ? "chevron-right" : "chevron-down"
                    iconSize: 10
                    iconColor: Theme.textMuted
                }

                MouseArea {
                    id: collapseMa
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: collapsed = !collapsed
                }
            }
        }
    }

    Item {
        id: contentArea
        anchors.top: wHeader.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 8
        visible: !collapsed
        opacity: collapsed ? 0 : 1
        Behavior on opacity { NumberAnimation { duration: Theme.animFast } }
    }

    /* Drag handle */
    MouseArea {
        id: dragArea
        anchors.fill: wHeader.visible ? wHeader : parent
        anchors.bottomMargin: wHeader.visible ? 0 : parent.height - 28
        property point pressPos
        onPressed: pressPos = Qt.point(mouse.x, mouse.y)
        onPositionChanged: {
            if (pressed) {
                widgetFrame.x += mouse.x - pressPos.x
                widgetFrame.y += mouse.y - pressPos.y
            }
        }
    }
}
