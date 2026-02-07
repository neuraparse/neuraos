import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Rectangle {
    id: wf
    width: 800; height: 500
    color: "transparent"

    property int windowId: -1
    property string windowTitle: "Window"
    property string windowIcon: "terminal"
    property color windowAccent: Theme.primary
    property bool isFocused: false
    property bool isMaximized: false
    property Item windowLayer: parent

    property real savedX: x
    property real savedY: y
    property real savedW: width
    property real savedH: height
    property bool isClosing: false

    signal closeRequested()
    signal minimizeRequested()
    signal maximizeRequested()
    signal focusRequested()
    signal closeAnimFinished()

    default property alias contentData: contentArea.data
    property alias contentItem: contentArea

    /* Open animation */
    scale: 0.92
    opacity: 0
    transformOrigin: Item.Center
    Component.onCompleted: openAnim.start()

    ParallelAnimation {
        id: openAnim
        NumberAnimation { target: wf; property: "scale"; to: 1.0; duration: 200; easing.type: Easing.OutCubic }
        NumberAnimation { target: wf; property: "opacity"; to: 1.0; duration: 200; easing.type: Easing.OutCubic }
    }

    function animateClose() {
        if (isClosing) return
        isClosing = true
        closeAnim.start()
    }
    ParallelAnimation {
        id: closeAnim
        NumberAnimation { target: wf; property: "scale"; to: 0.94; duration: 150; easing.type: Easing.InCubic }
        NumberAnimation { target: wf; property: "opacity"; to: 0; duration: 150; easing.type: Easing.InCubic }
        onStopped: wf.closeAnimFinished()
    }

    function animateMinimize() { minimizeAnim.start() }
    ParallelAnimation {
        id: minimizeAnim
        NumberAnimation { target: wf; property: "scale"; to: 0.8; duration: 180; easing.type: Easing.InCubic }
        NumberAnimation { target: wf; property: "opacity"; to: 0; duration: 180; easing.type: Easing.InCubic }
        onStopped: { wf.visible = false; wf.scale = 1.0; wf.opacity = 1.0 }
    }

    function animateRestore() {
        wf.scale = 0.8; wf.opacity = 0; wf.visible = true
        restoreAnim.start()
    }
    ParallelAnimation {
        id: restoreAnim
        NumberAnimation { target: wf; property: "scale"; to: 1.0; duration: 200; easing.type: Easing.OutCubic }
        NumberAnimation { target: wf; property: "opacity"; to: 1.0; duration: 200; easing.type: Easing.OutCubic }
    }

    /* ─── Shadow (3-layer, depth effect) ─── */
    Rectangle {
        visible: !isMaximized
        anchors.fill: frame; anchors.margins: -8
        radius: Theme.windowRadius + 8
        color: Qt.rgba(0, 0, 0, isFocused ? 0.18 : 0.06)
        Behavior on color { ColorAnimation { duration: Theme.animFast } }
    }
    Rectangle {
        visible: !isMaximized
        anchors.fill: frame; anchors.margins: -4
        radius: Theme.windowRadius + 4
        color: Qt.rgba(0, 0, 0, isFocused ? 0.12 : 0.04)
        Behavior on color { ColorAnimation { duration: Theme.animFast } }
    }
    Rectangle {
        visible: !isMaximized
        anchors.fill: frame; anchors.margins: -1
        radius: Theme.windowRadius + 1
        color: Qt.rgba(0, 0, 0, isFocused ? 0.06 : 0.02)
        Behavior on color { ColorAnimation { duration: Theme.animFast } }
    }

    /* ─── Main frame ─── */
    Rectangle {
        id: frame
        anchors.fill: parent
        anchors.margins: isMaximized ? 0 : 4
        radius: isMaximized ? 0 : Theme.windowRadius
        color: Theme.windowBg
        border.width: 1
        border.color: isFocused ? Theme.windowBorderFocused : Theme.windowBorder
        clip: true
        Behavior on border.color { ColorAnimation { duration: Theme.animFast } }

        /* Focus accent gradient line at top */
        Rectangle {
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.leftMargin: isMaximized ? 0 : Theme.windowRadius
            anchors.rightMargin: isMaximized ? 0 : Theme.windowRadius
            height: isFocused ? 2 : 0
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: Theme.primary }
                GradientStop { position: 1.0; color: Theme.secondary }
            }
            Behavior on height { NumberAnimation { duration: Theme.animFast } }
        }

        /* ─── Title bar ─── */
        Rectangle {
            id: titleBar
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: Theme.windowTitleH
            color: isFocused ? Theme.windowTitleBarFocused : Theme.windowTitleBar
            radius: isMaximized ? 0 : Theme.windowRadius
            Behavior on color { ColorAnimation { duration: Theme.animFast } }

            /* Square off bottom corners of title bar */
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: Theme.windowRadius
                color: parent.color
            }

            /* ─── Drag area (full title bar, buttons on top) ─── */
            MouseArea {
                id: dragArea
                anchors.fill: parent
                property point pressPos
                onPressed: { pressPos = Qt.point(mouse.x, mouse.y); wf.focusRequested() }
                onPositionChanged: {
                    if (pressed && !isMaximized) {
                        wf.x += mouse.x - pressPos.x
                        wf.y += mouse.y - pressPos.y
                    }
                }
                onDoubleClicked: wf.maximizeRequested()
            }

            /* ─── LEFT: Icon + Title ─── */
            Components.CanvasIcon {
                id: titleIcon
                x: 12
                y: (titleBar.height - iconSize) / 2
                iconName: wf.windowIcon
                iconColor: wf.windowAccent
                iconSize: 15
            }

            Text {
                id: titleText
                x: titleIcon.x + titleIcon.width + 8
                y: (titleBar.height - height) / 2
                width: titleBar.width - x - btnRow.width - 8
                text: wf.windowTitle
                color: isFocused ? Theme.text : Theme.textDim
                font.pixelSize: 13
                font.family: Theme.fontFamily
                elide: Text.ElideRight
                Behavior on color { ColorAnimation { duration: Theme.animFast } }
            }

            /* ─── RIGHT: Window Buttons (Rectangle-drawn, no font dependency) ─── */
            Row {
                id: btnRow
                anchors.right: parent.right
                anchors.rightMargin: isMaximized ? 0 : 2
                anchors.top: parent.top
                height: titleBar.height
                z: 10
                spacing: 0

                /* ── Minimize: horizontal line ── */
                Rectangle {
                    width: 46; height: titleBar.height
                    color: wfMinMa.containsMouse ? Qt.rgba(1, 1, 1, 0.12) : "transparent"

                    Rectangle {
                        anchors.centerIn: parent
                        width: 12; height: 1
                        color: isFocused ? "#FFFFFF" : "#8B8FA2"
                    }

                    MouseArea {
                        id: wfMinMa; anchors.fill: parent
                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: wf.minimizeRequested()
                    }
                }

                /* ── Maximize / Restore: square outline ── */
                Rectangle {
                    width: 46; height: titleBar.height
                    color: wfMaxMa.containsMouse ? Qt.rgba(1, 1, 1, 0.12) : "transparent"

                    Rectangle {
                        anchors.centerIn: parent
                        width: 10; height: 10
                        color: "transparent"
                        border.width: 1
                        border.color: isFocused ? "#FFFFFF" : "#8B8FA2"
                        visible: !isMaximized
                    }

                    /* Restore icon: two overlapping squares */
                    Item {
                        anchors.centerIn: parent
                        width: 12; height: 12
                        visible: isMaximized

                        Rectangle {
                            x: 2; y: 0; width: 8; height: 8
                            color: "transparent"; border.width: 1
                            border.color: isFocused ? "#FFFFFF" : "#8B8FA2"
                        }
                        Rectangle {
                            x: 0; y: 2; width: 8; height: 8
                            color: titleBar.color; border.width: 1
                            border.color: isFocused ? "#FFFFFF" : "#8B8FA2"
                        }
                    }

                    MouseArea {
                        id: wfMaxMa; anchors.fill: parent
                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: wf.maximizeRequested()
                    }
                }

                /* ── Close: X from two rotated rectangles ── */
                Rectangle {
                    width: 46; height: titleBar.height
                    color: wfCloseMa.containsMouse ? "#E81123" : "transparent"

                    Item {
                        anchors.centerIn: parent
                        width: 14; height: 14

                        Rectangle {
                            anchors.centerIn: parent
                            width: 14; height: 1.2
                            antialiasing: true; rotation: 45
                            color: wfCloseMa.containsMouse ? "#FFFFFF" : (isFocused ? "#FFFFFF" : "#8B8FA2")
                        }
                        Rectangle {
                            anchors.centerIn: parent
                            width: 14; height: 1.2
                            antialiasing: true; rotation: -45
                            color: wfCloseMa.containsMouse ? "#FFFFFF" : (isFocused ? "#FFFFFF" : "#8B8FA2")
                        }
                    }

                    MouseArea {
                        id: wfCloseMa; anchors.fill: parent
                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: wf.closeRequested()
                    }
                }
            }
        }

        Item {
            id: contentArea
            anchors.top: titleBar.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            clip: true
        }
    }

    /* ─── Resize handles ─── */
    MouseArea {
        anchors.right: parent.right; anchors.top: parent.top; anchors.bottom: parent.bottom; width: 5
        cursorShape: Qt.SizeHorCursor; enabled: !isMaximized
        property real startX; property real startW
        onPressed: { startX = mouseX; startW = wf.width; wf.focusRequested() }
        onPositionChanged: { if (pressed) wf.width = Math.max(Theme.windowMinW, startW + mouseX - startX) }
    }
    MouseArea {
        anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right; height: 5
        cursorShape: Qt.SizeVerCursor; enabled: !isMaximized
        property real startY; property real startH
        onPressed: { startY = mouseY; startH = wf.height; wf.focusRequested() }
        onPositionChanged: { if (pressed) wf.height = Math.max(Theme.windowMinH, startH + mouseY - startY) }
    }
    MouseArea {
        anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom; width: 5
        cursorShape: Qt.SizeHorCursor; enabled: !isMaximized
        property real startX; property real startW; property real startWx
        onPressed: { startX = mouseX; startW = wf.width; startWx = wf.x; wf.focusRequested() }
        onPositionChanged: { if (pressed) { var dx = mouseX - startX; var nw = startW - dx; if (nw >= Theme.windowMinW) { wf.width = nw; wf.x = startWx + dx } } }
    }
    MouseArea {
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; height: 5
        cursorShape: Qt.SizeVerCursor; enabled: !isMaximized
        property real startY; property real startH; property real startWy
        onPressed: { startY = mouseY; startH = wf.height; startWy = wf.y; wf.focusRequested() }
        onPositionChanged: { if (pressed) { var dy = mouseY - startY; var nh = startH - dy; if (nh >= Theme.windowMinH) { wf.height = nh; wf.y = startWy + dy } } }
    }
    MouseArea {
        anchors.right: parent.right; anchors.bottom: parent.bottom; width: 12; height: 12
        cursorShape: Qt.SizeFDiagCursor; enabled: !isMaximized
        property real sx; property real sy; property real sw; property real sh
        onPressed: { sx = mouseX; sy = mouseY; sw = wf.width; sh = wf.height; wf.focusRequested() }
        onPositionChanged: { if (pressed) { wf.width = Math.max(Theme.windowMinW, sw + mouseX - sx); wf.height = Math.max(Theme.windowMinH, sh + mouseY - sy) } }
    }

    /* Click catcher: consumes clicks so they DON'T pass to windows below */
    MouseArea { anchors.fill: parent; z: -1; onPressed: wf.focusRequested() }

    Behavior on x { enabled: isMaximized; NumberAnimation { duration: Theme.animFast } }
    Behavior on y { enabled: isMaximized; NumberAnimation { duration: Theme.animFast } }
    Behavior on width { enabled: isMaximized; NumberAnimation { duration: Theme.animFast } }
    Behavior on height { enabled: isMaximized; NumberAnimation { duration: Theme.animFast } }
}
