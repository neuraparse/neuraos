import QtQuick 2.15
import QtQuick.Layouts 1.15
import "../.."
import "../../components" as Components

Components.DesktopWidgetFrame {
    width: 210; height: 140
    widgetTitle: "Quick Launch"

    signal launchApp(string source)

    GridLayout {
        anchors.fill: parent
        columns: 4
        rowSpacing: 4
        columnSpacing: 4

        Repeater {
            model: ListModel {
                ListElement { label: "Term";   ico: "terminal"; clr: "#10B981"; src: "TerminalApp.qml" }
                ListElement { label: "Files";  ico: "folder";   clr: "#3B82F6"; src: "FileManagerApp.qml" }
                ListElement { label: "Mon";    ico: "monitor";  clr: "#F59E0B"; src: "SystemMonitorApp.qml" }
                ListElement { label: "AI";     ico: "robot";    clr: "#5B9AFF"; src: "AIAssistantApp.qml" }
                ListElement { label: "Store";  ico: "apps";     clr: "#7C3AED"; src: "AppStore.qml" }
                ListElement { label: "Net";    ico: "wifi";     clr: "#06B6D4"; src: "NetworkCenter.qml" }
                ListElement { label: "Config"; ico: "gear";     clr: "#9575F0"; src: "SettingsApp.qml" }
                ListElement { label: "Tasks";  ico: "dashboard"; clr: "#EF4444"; src: "TaskManagerApp.qml" }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: Theme.radiusTiny
                color: qaMa.containsMouse ? Theme.glassHover : "transparent"
                Behavior on color { ColorAnimation { duration: Theme.animFast } }

                Column {
                    anchors.centerIn: parent
                    spacing: 3

                    Components.CanvasIcon {
                        anchors.horizontalCenter: parent.horizontalCenter
                        iconName: model.ico
                        iconColor: model.clr
                        iconSize: 18
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: model.label; font.pixelSize: 9; color: Theme.textDim
                        font.family: Theme.fontFamily
                    }
                }

                MouseArea {
                    id: qaMa; anchors.fill: parent
                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: launchApp(model.src)
                }
            }
        }
    }
}
