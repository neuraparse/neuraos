import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: automationApp
    anchors.fill: parent

    property int selectedWorkflow: -1

    function getWorkflowDetail() {
        if (selectedWorkflow < 0) return null
        return Automation.getWorkflowDetail(selectedWorkflow)
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Top Bar ─── */
            Rectangle {
                Layout.fillWidth: true; height: 46; color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 10; spacing: 12

                    Text { text: "Automation Studio"; color: Theme.automation; font.pixelSize: 15; font.bold: true }

                    Item { Layout.fillWidth: true }

                    /* Recording indicator */
                    Rectangle {
                        width: recRow.width + 16; height: 26; radius: 13
                        color: Automation.isRecording ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.2) : "transparent"
                        visible: Automation.isRecording

                        Row {
                            id: recRow; anchors.centerIn: parent; spacing: 6
                            Rectangle {
                                width: 8; height: 8; radius: 4; color: Theme.error
                                SequentialAnimation on opacity {
                                    loops: Animation.Infinite
                                    NumberAnimation { to: 0.3; duration: 500 }
                                    NumberAnimation { to: 1.0; duration: 500 }
                                }
                            }
                            Text { text: "Recording"; color: Theme.error; font.pixelSize: 11; font.bold: true }
                        }
                    }

                    Row {
                        spacing: 12
                        StatPill { label: "Workflows"; value: Automation.workflowCount.toString(); pillColor: Theme.automation }
                    }
                }
            }

            /* ─── Content ─── */
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true; spacing: 0

                /* ─── Workflow List (left) ─── */
                Rectangle {
                    Layout.preferredWidth: 280; Layout.fillHeight: true; color: Theme.surface

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 8

                        Text { text: "Workflows"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                        ListView {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            model: Automation.workflows; clip: true; spacing: 4

                            delegate: Rectangle {
                                width: parent ? parent.width : 0
                                height: 68; radius: Theme.radiusTiny
                                color: selectedWorkflow === modelData.id ? Theme.glassActive :
                                       wfMa.containsMouse ? Theme.glassHover : "transparent"

                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 10; spacing: 3

                                    RowLayout {
                                        spacing: 8
                                        Rectangle {
                                            width: 8; height: 8; radius: 4
                                            color: modelData.status === "running" ? Theme.success :
                                                   modelData.status === "recording" ? Theme.error :
                                                   modelData.status === "completed" ? Theme.primary : Theme.textMuted
                                        }
                                        Text {
                                            text: modelData.name; color: Theme.text
                                            font.pixelSize: 12; font.bold: true
                                            Layout.fillWidth: true; elide: Text.ElideRight
                                        }
                                    }
                                    Text {
                                        text: modelData.stepCount + " steps | Run " + modelData.runCount + "x"
                                        color: Theme.textDim; font.pixelSize: 10
                                    }
                                    Text {
                                        text: "Last: " + modelData.lastRun
                                        color: Theme.textMuted; font.pixelSize: 9
                                    }
                                }

                                MouseArea {
                                    id: wfMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: selectedWorkflow = modelData.id
                                }
                            }
                        }

                        /* Create button */
                        Rectangle {
                            Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                            color: newWfMa.containsMouse ? Qt.darker(Theme.automation, 1.2) : Theme.automation

                            Text {
                                anchors.centerIn: parent
                                text: "+ New Workflow"; color: "white"; font.pixelSize: 12; font.bold: true
                            }
                            MouseArea {
                                id: newWfMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    var id = Automation.createWorkflow("Workflow " + (Automation.workflowCount + 1), "Custom workflow")
                                    selectedWorkflow = id
                                }
                            }
                        }
                    }
                }

                Rectangle { Layout.fillHeight: true; width: 1; color: Theme.glassBorder }

                /* ─── Workflow Detail (right) ─── */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true; color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 12

                        property var detail: getWorkflowDetail()

                        Text {
                            text: parent.detail ? parent.detail.name : "Select a Workflow"
                            color: Theme.text; font.pixelSize: 16; font.bold: true
                        }

                        Text {
                            text: parent.detail ? parent.detail.description : "Choose a workflow from the left panel to view details"
                            color: Theme.textDim; font.pixelSize: 12
                        }

                        /* Steps */
                        Rectangle {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            radius: Theme.radiusSmall; color: Theme.surface
                            border.width: 1; border.color: Theme.glassBorder
                            visible: parent.detail !== null

                            ColumnLayout {
                                anchors.fill: parent; anchors.margins: 12; spacing: 6

                                Text { text: "Steps"; color: Theme.text; font.pixelSize: 13; font.bold: true }

                                ListView {
                                    Layout.fillWidth: true; Layout.fillHeight: true
                                    model: parent.parent.parent.detail ? parent.parent.parent.detail.steps : []
                                    clip: true; spacing: 4

                                    delegate: Rectangle {
                                        width: parent ? parent.width : 0
                                        height: 44; radius: Theme.radiusTiny
                                        color: Theme.surfaceAlt

                                        RowLayout {
                                            anchors.fill: parent; anchors.margins: 10; spacing: 10

                                            Rectangle {
                                                width: 24; height: 24; radius: 12
                                                color: Qt.rgba(Theme.automation.r, Theme.automation.g, Theme.automation.b, 0.2)
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: (index + 1).toString()
                                                    color: Theme.automation; font.pixelSize: 11; font.bold: true
                                                }
                                            }

                                            Text {
                                                text: modelData.type; color: Theme.text
                                                font.pixelSize: 12; font.bold: true
                                            }

                                            Text {
                                                text: {
                                                    var p = modelData.params
                                                    if (p.app) return p.app
                                                    if (p.text) return '"' + p.text + '"'
                                                    if (p.seconds) return p.seconds + 's'
                                                    if (p.model) return p.model
                                                    return ""
                                                }
                                                color: Theme.textDim; font.pixelSize: 11
                                                Layout.fillWidth: true; elide: Text.ElideRight
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /* Action Buttons */
                        RowLayout {
                            Layout.fillWidth: true; spacing: 8
                            visible: parent.detail !== null

                            Rectangle {
                                Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                                color: playMa.containsMouse ? Qt.darker(Theme.success, 1.2) : Theme.success

                                Text { anchors.centerIn: parent; text: "Play"; color: "white"; font.pixelSize: 12; font.bold: true }
                                MouseArea {
                                    id: playMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: if (selectedWorkflow > 0) Automation.playWorkflow(selectedWorkflow)
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                                color: recMa.containsMouse ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.3) : Theme.surfaceAlt

                                Text {
                                    anchors.centerIn: parent
                                    text: Automation.isRecording ? "Stop Recording" : "Record"
                                    color: Theme.error; font.pixelSize: 12; font.bold: true
                                }
                                MouseArea {
                                    id: recMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (Automation.isRecording) Automation.stopRecording()
                                        else if (selectedWorkflow > 0) Automation.startRecording(selectedWorkflow)
                                    }
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true; height: 36; radius: Theme.radiusSmall
                                color: delMa.containsMouse ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.2) : "transparent"
                                border.width: 1; border.color: Theme.error

                                Text { anchors.centerIn: parent; text: "Delete"; color: Theme.error; font.pixelSize: 12 }
                                MouseArea {
                                    id: delMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (selectedWorkflow > 0) {
                                            Automation.deleteWorkflow(selectedWorkflow)
                                            selectedWorkflow = -1
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

    component StatPill: Rectangle {
        property string label: ""; property string value: ""; property color pillColor: Theme.primary
        width: spRow.width + 16; height: 24; radius: 12
        color: Qt.rgba(pillColor.r, pillColor.g, pillColor.b, 0.12)
        Row { id: spRow; anchors.centerIn: parent; spacing: 4
            Text { text: parent.parent.label; color: Theme.textDim; font.pixelSize: 10 }
            Text { text: parent.parent.value; color: parent.parent.pillColor; font.pixelSize: 10; font.bold: true }
        }
    }
}
