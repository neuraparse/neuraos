import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: quantumApp
    anchors.fill: parent

    property int numQubits: 3
    property var circuit: []
    property var results: []
    property bool simRunning: false

    Component.onCompleted: resetCircuit()

    function resetCircuit() {
        circuit = []
        for (var i = 0; i < numQubits; i++) {
            circuit.push([])
        }
        results = []
        circuitChanged()
    }

    function addGate(qubit, gate) {
        var c = circuit
        c[qubit].push(gate)
        circuit = c
        circuitChanged()
    }

    function simulate() {
        simRunning = true
        /* Simulated quantum measurement results */
        var res = []
        var states = Math.pow(2, numQubits)
        var total = 1000
        var remaining = total

        for (var i = 0; i < states; i++) {
            var binary = ""
            for (var b = numQubits - 1; b >= 0; b--) {
                binary += ((i >> b) & 1).toString()
            }

            var prob
            if (i === states - 1) {
                prob = remaining
            } else {
                prob = Math.floor(Math.random() * remaining * 0.5)
                remaining -= prob
            }

            res.push({ state: "|" + binary + "\u27E9", count: prob, probability: prob / total })
        }

        results = res
        resultsChanged()
        simRunning = false
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Toolbar */
            Rectangle {
                Layout.fillWidth: true; height: 42
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.margins: 8; spacing: 8

                    Text { text: "Quantum Circuit Lab"; color: Theme.text; font.pixelSize: 13; font.bold: true }
                    Item { Layout.fillWidth: true }

                    Text { text: "Qubits:"; color: Theme.textDim; font.pixelSize: 10 }

                    Repeater {
                        model: [2, 3, 4, 5]

                        Rectangle {
                            width: 28; height: 24; radius: Theme.radiusSmall
                            color: numQubits === modelData ? Theme.primary : qbMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Text { anchors.centerIn: parent; text: modelData.toString(); color: numQubits === modelData ? "#000" : Theme.text; font.pixelSize: 10 }
                            MouseArea { id: qbMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { numQubits = modelData; resetCircuit() } }
                        }
                    }

                    Rectangle { width: 1; height: 20; color: Theme.surfaceLight }

                    Rectangle {
                        width: 60; height: 26; radius: Theme.radiusSmall
                        color: Theme.surfaceAlt
                        Text { anchors.centerIn: parent; text: "Reset"; color: Theme.warning; font.pixelSize: 10 }
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: resetCircuit() }
                    }

                    Rectangle {
                        width: 80; height: 26; radius: Theme.radiusSmall
                        color: simMa.containsMouse ? Qt.darker(Theme.primary, 1.2) : Theme.primary
                        Text { anchors.centerIn: parent; text: "Simulate"; color: "#000"; font.bold: true; font.pixelSize: 10 }
                        MouseArea { id: simMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: simulate() }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* Circuit editor */
                ColumnLayout {
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width * 0.6
                    spacing: 0

                    /* Gate palette */
                    Rectangle {
                        Layout.fillWidth: true; height: 50
                        color: Theme.surface

                        Row {
                            anchors.centerIn: parent; spacing: 6

                            Text { text: "Gates:"; color: Theme.textDim; font.pixelSize: 10; anchors.verticalCenter: parent.verticalCenter }

                            Repeater {
                                model: [
                                    { name: "H", color: "#00D9FF" },
                                    { name: "X", color: "#EF4444" },
                                    { name: "Y", color: "#10B981" },
                                    { name: "Z", color: "#3B82F6" },
                                    { name: "T", color: "#F59E0B" },
                                    { name: "S", color: "#A78BFA" },
                                    { name: "CNOT", color: "#EC4899" }
                                ]

                                Components.QuantumGate {
                                    width: modelData.name === "CNOT" ? 56 : 38; height: 34
                                    gateName: modelData.name
                                    gateColor: modelData.color
                                    isActive: false
                                    onClicked: {
                                        /* Add to first qubit for demo */
                                        for (var q = 0; q < numQubits; q++) {
                                            if (circuit[q].length <= 8) {
                                                addGate(q, modelData.name)
                                                break
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                    /* Circuit visualization */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: Theme.darkMode ? "#0A0A14" : "#F2F3F8"

                        Canvas {
                            id: circuitCanvas
                            anchors.fill: parent; anchors.margins: 10

                            property var circuitData: circuit
                            onCircuitDataChanged: requestPaint()

                            onPaint: {
                                var ctx = getContext("2d")
                                ctx.clearRect(0, 0, width, height)

                                var lineSpacing = height / (numQubits + 1)
                                var gateSize = 36
                                var gateSpacing = 50

                                /* Qubit lines */
                                for (var q = 0; q < numQubits; q++) {
                                    var y = lineSpacing * (q + 1)

                                    /* Label */
                                    ctx.fillStyle = "#A78BFA"
                                    ctx.font = "bold 11px monospace"
                                    ctx.fillText("|q" + q + "\u27E9", 4, y + 4)

                                    /* Wire */
                                    ctx.beginPath()
                                    ctx.moveTo(50, y)
                                    ctx.lineTo(width - 10, y)
                                    ctx.strokeStyle = Qt.rgba(0.66, 0.55, 0.98, 0.2)
                                    ctx.lineWidth = 1
                                    ctx.stroke()

                                    /* Gates on this qubit */
                                    var gates = circuitData[q] || []
                                    for (var g = 0; g < gates.length; g++) {
                                        var gx = 70 + g * gateSpacing
                                        var gy = y - gateSize / 2

                                        /* Gate box */
                                        ctx.fillStyle = Qt.rgba(0.66, 0.55, 0.98, 0.15)
                                        ctx.strokeStyle = "#A78BFA"
                                        ctx.lineWidth = 1
                                        ctx.beginPath()
                                        var rr = 4
                                        ctx.moveTo(gx + rr, gy)
                                        ctx.lineTo(gx + gateSize - rr, gy)
                                        ctx.arcTo(gx + gateSize, gy, gx + gateSize, gy + rr, rr)
                                        ctx.lineTo(gx + gateSize, gy + gateSize - rr)
                                        ctx.arcTo(gx + gateSize, gy + gateSize, gx + gateSize - rr, gy + gateSize, rr)
                                        ctx.lineTo(gx + rr, gy + gateSize)
                                        ctx.arcTo(gx, gy + gateSize, gx, gy + gateSize - rr, rr)
                                        ctx.lineTo(gx, gy + rr)
                                        ctx.arcTo(gx, gy, gx + rr, gy, rr)
                                        ctx.fill()
                                        ctx.stroke()

                                        /* Gate label */
                                        ctx.fillStyle = Theme.darkMode ? "#FFFFFF" : "#1A1A2E"
                                        ctx.font = "bold 12px monospace"
                                        ctx.textAlign = "center"
                                        ctx.fillText(gates[g], gx + gateSize / 2, y + 4)
                                        ctx.textAlign = "start"
                                    }

                                    /* Measurement at end */
                                    var mx = width - 40
                                    ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.3)
                                    ctx.beginPath()
                                    ctx.arc(mx, y, 10, 0, Math.PI, true)
                                    ctx.stroke()
                                    ctx.beginPath()
                                    ctx.moveTo(mx, y)
                                    ctx.lineTo(mx + 6, y - 10)
                                    ctx.stroke()
                                }
                            }
                        }
                    }
                }

                Rectangle { width: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

                /* Results panel */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: Theme.background

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 10; spacing: 8

                        Text { text: "Measurement Results"; color: Theme.text; font.pixelSize: 12; font.bold: true }
                        Text { text: "1000 shots"; color: Theme.textDim; font.pixelSize: 10 }

                        /* Histogram */
                        Flickable {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            contentHeight: resultCol.height
                            clip: true

                            Column {
                                id: resultCol
                                width: parent.width
                                spacing: 4

                                Repeater {
                                    model: results.length

                                    RowLayout {
                                        width: parent.width; spacing: 6

                                        Text {
                                            text: results[index] ? results[index].state : ""
                                            color: "#A78BFA"; font.pixelSize: 11; font.family: "Liberation Mono"
                                            Layout.preferredWidth: 50
                                        }

                                        Rectangle {
                                            Layout.fillWidth: true; height: 16; radius: 3
                                            color: Theme.surfaceLight

                                            Rectangle {
                                                width: parent.width * (results[index] ? results[index].probability : 0)
                                                height: parent.height; radius: 3
                                                color: "#A78BFA"
                                            }
                                        }

                                        Text {
                                            text: results[index] ? results[index].count.toString() : ""
                                            color: Theme.text; font.pixelSize: 10
                                            Layout.preferredWidth: 35
                                            horizontalAlignment: Text.AlignRight
                                        }
                                    }
                                }

                                /* Empty state */
                                Text {
                                    visible: results.length === 0
                                    text: "Add gates and click\n'Simulate' to see results"
                                    color: Theme.textDim; font.pixelSize: 11
                                    horizontalAlignment: Text.AlignHCenter
                                    width: parent.width
                                    topPadding: 40
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
