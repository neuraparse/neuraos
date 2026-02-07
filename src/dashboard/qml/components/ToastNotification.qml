import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."

Rectangle {
    id: toast
    width: 320; height: 48
    radius: Theme.radiusSmall
    color: Theme.glass
    border.width: 1
    border.color: Theme.glassBorder
    opacity: 0; y: -60
    visible: opacity > 0

    property string message: ""
    property string variant: "info"
    property color iconColor: variant === "error" ? Theme.error : variant === "success" ? Theme.success : Theme.primary

    function show(msg, type) {
        message = msg
        if (type) variant = type
        showAnim.start()
        dismissTimer.restart()
    }

    ParallelAnimation {
        id: showAnim
        NumberAnimation { target: toast; property: "opacity"; to: 1; duration: 200; easing.type: Easing.OutQuint }
        NumberAnimation { target: toast; property: "y"; to: 12; duration: 250; easing.type: Easing.OutQuint }
    }
    ParallelAnimation {
        id: hideAnim
        NumberAnimation { target: toast; property: "opacity"; to: 0; duration: 200; easing.type: Easing.InQuint }
        NumberAnimation { target: toast; property: "y"; to: -60; duration: 250; easing.type: Easing.InQuint }
    }

    Timer { id: dismissTimer; interval: 3000; onTriggered: hideAnim.start() }

    RowLayout {
        anchors.fill: parent; anchors.margins: 12; spacing: 10
        Rectangle { width: 4; height: 24; radius: 2; color: toast.iconColor }
        Text { Layout.fillWidth: true; text: toast.message; color: Theme.text; font.pixelSize: 12; font.family: Theme.fontFamily; elide: Text.ElideRight }
    }
}
