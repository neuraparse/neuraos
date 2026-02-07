import QtQuick 2.15
import QtQuick.Layouts 1.15
import "."

Rectangle {
    id: dock
    color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.95)

    property int currentIndex: 0
    signal pageSelected(int index)

    /* Top border glow */
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 1
        color: Theme.surfaceLight
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 20
        anchors.rightMargin: 20
        spacing: 0

        Repeater {
            model: ListModel {
                ListElement { label: "Home";     icon: "\u2302"; page: 0 }
                ListElement { label: "AI";       icon: "\u2699"; page: 1 }
                ListElement { label: "Monitor";  icon: "\u2261"; page: 2 }
                ListElement { label: "Terminal"; icon: "\u2756"; page: 3 }
                ListElement { label: "Files";    icon: "\u2750"; page: 4 }
                ListElement { label: "Settings"; icon: "\u2731"; page: 5 }
            }

            delegate: Item {
                Layout.fillWidth: true
                Layout.fillHeight: true

                property bool isActive: dock.currentIndex === model.page

                MouseArea {
                    anchors.fill: parent
                    onClicked: dock.pageSelected(model.page)
                    hoverEnabled: true

                    Column {
                        anchors.centerIn: parent
                        spacing: 2

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: model.icon
                            font.pixelSize: 22
                            color: isActive ? Theme.primary : Theme.textDim

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: model.label
                            font.pixelSize: 10
                            font.bold: isActive
                            color: isActive ? Theme.primary : Theme.textDim

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }
                        }

                        /* Active indicator dot */
                        Rectangle {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: isActive ? 16 : 0
                            height: 3
                            radius: 2
                            color: Theme.primary
                            opacity: isActive ? 1 : 0

                            Behavior on width { NumberAnimation { duration: Theme.animNormal } }
                            Behavior on opacity { NumberAnimation { duration: Theme.animNormal } }
                        }
                    }
                }
            }
        }
    }
}
