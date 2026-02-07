import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."

Item {
    id: termPage

    Rectangle {
        anchors.fill: parent
        anchors.margins: 12
        radius: Theme.radius
        color: "#0C0C0C"
        border.width: 1
        border.color: Theme.surfaceLight

        /* Terminal header */
        Rectangle {
            id: termHeader
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 36
            color: Theme.surface
            radius: Theme.radius

            /* Fix bottom corners */
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: Theme.radius
                color: Theme.surface
            }

            Row {
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                anchors.leftMargin: 12
                spacing: 8

                /* Traffic light dots */
                Rectangle { width: 10; height: 10; radius: 5; color: "#FF5F57" }
                Rectangle { width: 10; height: 10; radius: 5; color: "#FFBD2E" }
                Rectangle { width: 10; height: 10; radius: 5; color: "#28CA41" }

                Text {
                    text: "  neuraos ~ /bin/sh"
                    color: Theme.textDim
                    font.pixelSize: Theme.fontSizeSmall
                    font.family: "Liberation Mono"
                }
            }
        }

        /* Terminal output area */
        Flickable {
            id: termFlick
            anchors.top: termHeader.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: inputRow.top
            anchors.margins: 8
            contentHeight: termOutput.height
            clip: true

            TextEdit {
                id: termOutput
                width: parent.width
                readOnly: true
                color: "#00FF00"
                font.family: "Liberation Mono"
                font.pixelSize: 13
                wrapMode: TextEdit.Wrap
                selectByMouse: true
                selectionColor: Theme.primary
                text: "NeuralOS v2.0.0 - AI-Native Embedded Linux\n" +
                      "Kernel: " + SystemInfo.kernelVersion + "\n" +
                      "Host: " + SystemInfo.hostname + "\n" +
                      "─────────────────────────────────────\n" +
                      "Type commands below. This is a lightweight shell.\n\n"
            }
        }

        /* Input row */
        Rectangle {
            id: inputRow
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: 36
            color: "#111111"

            Row {
                anchors.fill: parent
                anchors.leftMargin: 8
                anchors.rightMargin: 8
                spacing: 6

                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: "root@neuraos:~$"
                    color: "#00D9FF"
                    font.family: "Liberation Mono"
                    font.pixelSize: 13
                    font.bold: true
                }

                TextInput {
                    id: cmdInput
                    anchors.verticalCenter: parent.verticalCenter
                    width: parent.width - 140
                    color: "#00FF00"
                    font.family: "Liberation Mono"
                    font.pixelSize: 13
                    clip: true
                    focus: true
                    selectByMouse: true
                    selectionColor: Theme.primary

                    onAccepted: {
                        var cmd = text.trim()
                        if (cmd === "") return

                        termOutput.text += "root@neuraos:~$ " + cmd + "\n"

                        /* Built-in commands */
                        if (cmd === "clear") {
                            termOutput.text = ""
                        } else if (cmd === "help") {
                            termOutput.text += "Built-in: clear, help, uname, uptime, whoami, neofetch\n"
                        } else if (cmd === "uname" || cmd === "uname -a") {
                            termOutput.text += "Linux " + SystemInfo.hostname + " " + SystemInfo.kernelVersion + " aarch64 NeuralOS\n"
                        } else if (cmd === "uptime") {
                            termOutput.text += SystemInfo.uptime + "\n"
                        } else if (cmd === "whoami") {
                            termOutput.text += "root\n"
                        } else if (cmd === "neofetch") {
                            termOutput.text += "  _   _                      ___  ____\n"
                            termOutput.text += " | \\ | | ___ _   _ _ __ __ _/ _ \\/ ___|\n"
                            termOutput.text += " |  \\| |/ _ \\ | | | '__/ _` | | | \\___ \\\n"
                            termOutput.text += " | |\\  |  __/ |_| | | | (_| | |_| |___) |\n"
                            termOutput.text += " |_| \\_|\\___|\\__,_|_|  \\__,_|\\___/|____/\n"
                            termOutput.text += "  OS: NeuralOS 2.0.0\n"
                            termOutput.text += "  Kernel: " + SystemInfo.kernelVersion + "\n"
                            termOutput.text += "  Host: " + SystemInfo.hostname + "\n"
                            termOutput.text += "  Uptime: " + SystemInfo.uptime + "\n"
                            termOutput.text += "  CPU: " + Math.round(SystemInfo.cpuUsage) + "% usage\n"
                            termOutput.text += "  Memory: " + (SystemInfo.memoryUsed / 1048576).toFixed(0) + " / " + (SystemInfo.memoryTotal / 1048576).toFixed(0) + " MB\n"
                            termOutput.text += "  NPU: " + NPUMonitor.deviceCount + " device(s)\n"
                            termOutput.text += "  NPIE: " + NPIE.version + "\n"
                        } else {
                            termOutput.text += "sh: " + cmd + ": command not found\n"
                        }

                        text = ""
                        termFlick.contentY = termOutput.height - termFlick.height
                    }
                }
            }
        }
    }
}
