import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: calcApp
    anchors.fill: parent

    property string expression: ""
    property string display: "0"
    property bool newInput: true
    property string pendingOp: ""
    property double memory: 0

    function pressDigit(d) {
        if (newInput) { display = d; newInput = false }
        else { display = (display === "0") ? d : display + d }
    }

    function pressOp(op) {
        if (pendingOp !== "") calculate()
        memory = parseFloat(display)
        pendingOp = op
        expression = display + " " + op
        newInput = true
    }

    function calculate() {
        if (pendingOp === "") return
        var b = parseFloat(display)
        var result = 0
        if (pendingOp === "+") result = memory + b
        else if (pendingOp === "−") result = memory - b
        else if (pendingOp === "×") result = memory * b
        else if (pendingOp === "÷") result = (b !== 0) ? memory / b : 0
        expression = ""
        display = parseFloat(result.toFixed(10)).toString()
        pendingOp = ""
        newInput = true
    }

    function pressClear() {
        expression = ""; display = "0"; pendingOp = ""; memory = 0; newInput = true
    }

    function pressPercent() {
        display = (parseFloat(display) / 100).toString()
        newInput = true
    }

    function pressNegate() {
        if (display !== "0") {
            display = (parseFloat(display) * -1).toString()
        }
    }

    function pressDot() {
        if (newInput) { display = "0."; newInput = false }
        else if (display.indexOf(".") === -1) { display += "." }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 0
            spacing: 0

            /* Display Area */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 4

                    Item { Layout.fillHeight: true }

                    Text {
                        Layout.fillWidth: true
                        text: expression
                        font.pixelSize: 14
                        color: Theme.textDim
                        horizontalAlignment: Text.AlignRight
                    }

                    Text {
                        Layout.fillWidth: true
                        text: display
                        font.pixelSize: 42
                        font.weight: Font.Light
                        color: Theme.text
                        horizontalAlignment: Text.AlignRight
                        elide: Text.ElideRight
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: Theme.surfaceLight
            }

            /* Button Grid */
            GridLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: 8
                columns: 4
                rowSpacing: 6
                columnSpacing: 6

                /* Row 1: C, +/-, %, ÷ */
                CalcButton { label: "C"; bgColor: Theme.surfaceAlt; textColor: Theme.error; onClicked: pressClear() }
                CalcButton { label: "+/−"; bgColor: Theme.surfaceAlt; onClicked: pressNegate() }
                CalcButton { label: "%"; bgColor: Theme.surfaceAlt; onClicked: pressPercent() }
                CalcButton { label: "÷"; bgColor: Theme.primary; textColor: "#FFFFFF"; onClicked: pressOp("÷") }

                /* Row 2: 7 8 9 × */
                CalcButton { label: "7"; onClicked: pressDigit("7") }
                CalcButton { label: "8"; onClicked: pressDigit("8") }
                CalcButton { label: "9"; onClicked: pressDigit("9") }
                CalcButton { label: "×"; bgColor: Theme.primary; textColor: "#FFFFFF"; onClicked: pressOp("×") }

                /* Row 3: 4 5 6 − */
                CalcButton { label: "4"; onClicked: pressDigit("4") }
                CalcButton { label: "5"; onClicked: pressDigit("5") }
                CalcButton { label: "6"; onClicked: pressDigit("6") }
                CalcButton { label: "−"; bgColor: Theme.primary; textColor: "#FFFFFF"; onClicked: pressOp("−") }

                /* Row 4: 1 2 3 + */
                CalcButton { label: "1"; onClicked: pressDigit("1") }
                CalcButton { label: "2"; onClicked: pressDigit("2") }
                CalcButton { label: "3"; onClicked: pressDigit("3") }
                CalcButton { label: "+"; bgColor: Theme.primary; textColor: "#FFFFFF"; onClicked: pressOp("+") }

                /* Row 5: 0 (2-wide), ., = */
                CalcButton {
                    label: "0"
                    Layout.columnSpan: 2
                    onClicked: pressDigit("0")
                }
                CalcButton { label: "."; onClicked: pressDot() }
                CalcButton { label: "="; bgColor: Theme.success; textColor: "#FFFFFF"; onClicked: calculate() }
            }
        }
    }

    /* Inline CalcButton component */
    component CalcButton: Rectangle {
        property string label: ""
        property color bgColor: Theme.surface
        property color textColor: Theme.text
        signal clicked()

        Layout.fillWidth: true
        Layout.fillHeight: true
        radius: Theme.radiusSmall
        color: calcBtnMa.pressed ? Qt.darker(bgColor, 1.15) : calcBtnMa.containsMouse ? Qt.lighter(bgColor, 1.1) : bgColor

        Behavior on color { ColorAnimation { duration: 80 } }

        Text {
            anchors.centerIn: parent
            text: label
            font.pixelSize: 22
            font.weight: Font.Medium
            color: textColor
        }

        MouseArea {
            id: calcBtnMa
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: parent.clicked()
        }
    }
}
