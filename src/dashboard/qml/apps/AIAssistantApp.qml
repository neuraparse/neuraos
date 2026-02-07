import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: aiApp
    anchors.fill: parent

    /* ─── State ─── */
    property string selectedModel: "local"
    property int selectedConversation: 0
    property bool isThinking: false
    property int mockResponseIndex: 0

    property var mockResponses: [
        "I'll help you with that! Here's what I found...",
        "Based on my analysis...",
        "Here's a solution for your request...",
        "Let me break this down step by step..."
    ]

    property var conversations: [
        "Code Review Session",
        "Debug Python Error",
        "Explain Quantum Computing",
        "Write API Docs",
        "Optimize SQL Query",
        "Design System Help"
    ]

    /* ─── Message Model ─── */
    ListModel {
        id: messageModel

        Component.onCompleted: {
            append({
                isUser: false,
                sender: "NeuralOS AI",
                text: "Hello! I'm NeuralOS AI Assistant. I can help you with code, explanations, file management, and more. What would you like to work on?",
                time: "just now"
            })
            append({
                isUser: true,
                sender: "You",
                text: "Can you explain how NPU inference works?",
                time: "just now"
            })
            append({
                isUser: false,
                sender: "NeuralOS AI",
                text: "Neural Processing Units (NPUs) accelerate machine learning inference by executing tensor operations in dedicated hardware. In NeuralOS, the NPU pipeline works as follows:\n\n1. Model Loading \u2014 The .tflite or .onnx model is loaded into NPU memory\n2. Input Preprocessing \u2014 Data is converted to the model's expected tensor format\n3. Inference Execution \u2014 The NPU processes layers in parallel using its matrix cores\n4. Output Postprocessing \u2014 Results are converted back to application-readable format\n\nThe NeuralOS NPIE (Neural Processing Inference Engine) handles this entire pipeline transparently.",
                time: "just now"
            })
        }
    }

    /* ─── Thinking Steps Model ─── */
    ListModel {
        id: thinkingSteps
    }

    /* ─── Timers ─── */
    Timer {
        id: responseTimer
        interval: 1500
        repeat: false
        onTriggered: addAIResponse()
    }

    Timer {
        id: thinkingStep1Timer
        interval: 400
        repeat: false
        onTriggered: {
            if (thinkingSteps.count > 0) thinkingSteps.setProperty(0, "done", true)
            if (thinkingSteps.count > 1) thinkingSteps.setProperty(1, "active", true)
        }
    }

    Timer {
        id: thinkingStep2Timer
        interval: 900
        repeat: false
        onTriggered: {
            if (thinkingSteps.count > 1) thinkingSteps.setProperty(1, "done", true)
            if (thinkingSteps.count > 2) thinkingSteps.setProperty(2, "active", true)
        }
    }

    /* ─── Functions ─── */
    function sendMessage(text) {
        if (text.trim() === "") return
        messageModel.append({
            isUser: true,
            sender: "You",
            text: text.trim(),
            time: "just now"
        })
        msgInput.text = ""
        chatFlickable.contentY = chatFlickable.contentHeight - chatFlickable.height
        isThinking = true
        thinkingSteps.clear()
        thinkingSteps.append({ label: "Analyzing query...", done: false, active: true })
        thinkingSteps.append({ label: "Searching knowledge base...", done: false, active: false })
        thinkingSteps.append({ label: "Generating response...", done: false, active: false })
        thinkingStep1Timer.start()
        thinkingStep2Timer.start()
        responseTimer.start()
    }

    function addAIResponse() {
        var resp = mockResponses[mockResponseIndex % mockResponses.length]
        mockResponseIndex++
        messageModel.append({
            isUser: false,
            sender: "NeuralOS AI",
            text: resp,
            time: "just now"
        })
        isThinking = false
        thinkingSteps.clear()
        scrollToBottom.start()
    }

    function clearChat() {
        messageModel.clear()
        isThinking = false
        thinkingSteps.clear()
    }

    Timer {
        id: scrollToBottom
        interval: 50
        repeat: false
        onTriggered: {
            if (chatFlickable.contentHeight > chatFlickable.height)
                chatFlickable.contentY = chatFlickable.contentHeight - chatFlickable.height
        }
    }

    /* ─── Main Layout ─── */
    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* ═══════════════════════════════════════════════════
               LEFT SIDEBAR (240px)
               ═══════════════════════════════════════════════════ */
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 0

                    /* ── Header ── */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 48
                        color: "transparent"

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 14
                            anchors.rightMargin: 10
                            spacing: 8

                            Text {
                                Layout.fillWidth: true
                                text: "AI Assistant"
                                font.pixelSize: Theme.fontSizeLarge
                                font.bold: true
                                font.family: Theme.fontFamily
                                color: Theme.text
                            }

                            Rectangle {
                                width: 30; height: 30
                                radius: Theme.radiusSmall
                                color: newChatMa.containsMouse
                                    ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.18)
                                    : Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.10)

                                Behavior on color { ColorAnimation { duration: Theme.animFast } }

                                Components.CanvasIcon {
                                    anchors.centerIn: parent
                                    iconName: "plus"
                                    iconSize: 14
                                    iconColor: Theme.primary
                                }

                                MouseArea {
                                    id: newChatMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: clearChat()
                                }
                            }
                        }
                    }

                    /* ── Model Selector ── */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 40
                        Layout.leftMargin: 10
                        Layout.rightMargin: 10
                        color: "transparent"

                        Row {
                            anchors.centerIn: parent
                            spacing: 4

                            Repeater {
                                model: [
                                    { key: "local", label: "NeuralOS Local" },
                                    { key: "gpt4",  label: "GPT-4" },
                                    { key: "claude", label: "Claude" }
                                ]

                                Rectangle {
                                    width: modelPillLabel.implicitWidth + 16
                                    height: 26
                                    radius: 13
                                    color: selectedModel === modelData.key
                                        ? Theme.primary
                                        : modelPillMa.containsMouse
                                            ? Theme.surfaceAlt
                                            : "transparent"

                                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                                    Text {
                                        id: modelPillLabel
                                        anchors.centerIn: parent
                                        text: modelData.label
                                        font.pixelSize: 10
                                        font.bold: selectedModel === modelData.key
                                        font.family: Theme.fontFamily
                                        color: selectedModel === modelData.key ? "#FFFFFF" : Theme.textDim
                                    }

                                    MouseArea {
                                        id: modelPillMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: selectedModel = modelData.key
                                    }
                                }
                            }
                        }
                    }

                    /* ── Separator ── */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.leftMargin: 10
                        Layout.rightMargin: 10
                        height: 1
                        color: Theme.surfaceLight
                    }

                    Item { height: 6 }

                    /* ── Conversation List ── */
                    Flickable {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentHeight: convoCol.height
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded; width: 4
                            contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                        }

                        Column {
                            id: convoCol
                            width: parent.width
                            spacing: 2

                            Repeater {
                                model: conversations.length

                                Rectangle {
                                    width: parent.width - 8
                                    height: 40
                                    x: 4
                                    radius: Theme.radiusSmall
                                    color: index === selectedConversation
                                        ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.12)
                                        : convoItemMa.containsMouse
                                            ? Theme.surfaceAlt
                                            : "transparent"

                                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 10
                                        anchors.rightMargin: 10
                                        spacing: 8

                                        Components.CanvasIcon {
                                            iconName: "chat"
                                            iconSize: 14
                                            iconColor: index === selectedConversation
                                                ? Theme.primary
                                                : Theme.textDim
                                        }

                                        Text {
                                            Layout.fillWidth: true
                                            text: conversations[index]
                                            font.pixelSize: 12
                                            font.family: Theme.fontFamily
                                            color: index === selectedConversation
                                                ? Theme.text
                                                : Theme.textDim
                                            elide: Text.ElideRight
                                        }
                                    }

                                    MouseArea {
                                        id: convoItemMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: selectedConversation = index
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* ── Vertical Separator ── */
            Rectangle {
                Layout.preferredWidth: 1
                Layout.fillHeight: true
                color: Theme.surfaceLight
            }

            /* ═══════════════════════════════════════════════════
               RIGHT: CHAT AREA
               ═══════════════════════════════════════════════════ */
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0

                /* ── Chat Header ── */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 44
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 14
                        anchors.rightMargin: 14
                        spacing: 8

                        Components.CanvasIcon {
                            iconName: "robot"
                            iconSize: 16
                            iconColor: Theme.primary
                        }

                        Text {
                            Layout.fillWidth: true
                            text: selectedModel === "local" ? "NeuralOS Local Model"
                                : selectedModel === "gpt4" ? "GPT-4"
                                : "Claude"
                            font.pixelSize: 14
                            font.bold: true
                            font.family: Theme.fontFamily
                            color: Theme.text
                        }

                        /* Online indicator */
                        Row {
                            spacing: 5
                            Rectangle {
                                width: 6; height: 6; radius: 3
                                anchors.verticalCenter: parent.verticalCenter
                                color: Theme.success
                            }
                            Text {
                                text: "Online"
                                font.pixelSize: 10
                                font.family: Theme.fontFamily
                                color: Theme.success
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }

                        /* Clear button */
                        Rectangle {
                            width: 60; height: 28
                            radius: Theme.radiusTiny
                            color: clearBtnMa.containsMouse
                                ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.12)
                                : "transparent"

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }

                            Text {
                                anchors.centerIn: parent
                                text: "Clear"
                                font.pixelSize: 11
                                font.family: Theme.fontFamily
                                color: Theme.error
                            }

                            MouseArea {
                                id: clearBtnMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: clearChat()
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                /* ── Message List ── */
                Flickable {
                    id: chatFlickable
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentHeight: chatColumn.height + 20
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    visible: messageModel.count > 0

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded; width: 6
                        contentItem: Rectangle { implicitWidth: 6; radius: 3; color: Theme.textMuted; opacity: 0.4 }
                    }

                    Column {
                        id: chatColumn
                        width: parent.width
                        spacing: 4
                        topPadding: 12
                        bottomPadding: 12

                        Repeater {
                            model: messageModel

                            Item {
                                width: chatColumn.width
                                height: msgCol.height + 8

                                Column {
                                    id: msgCol
                                    anchors.left: model.isUser ? undefined : parent.left
                                    anchors.right: model.isUser ? parent.right : undefined
                                    anchors.leftMargin: model.isUser ? 0 : 16
                                    anchors.rightMargin: model.isUser ? 16 : 0
                                    spacing: 3
                                    width: Math.min(parent.width * 0.7, msgBubble.implicitWidth + 24)

                                    /* Sender label */
                                    Text {
                                        text: model.sender
                                        font.pixelSize: 10
                                        font.family: Theme.fontFamily
                                        font.bold: true
                                        color: Theme.textMuted
                                        anchors.left: model.isUser ? undefined : parent.left
                                        anchors.right: model.isUser ? parent.right : undefined
                                        anchors.leftMargin: model.isUser ? 0 : 4
                                        anchors.rightMargin: model.isUser ? 4 : 0
                                    }

                                    /* Message bubble */
                                    Rectangle {
                                        id: msgBubble
                                        width: parent.width
                                        height: msgText.implicitHeight + 24
                                        radius: 16
                                        color: model.isUser ? Theme.primary : Theme.surfaceAlt

                                        /* Corner mask: top-right flat for user, top-left flat for AI */
                                        Rectangle {
                                            visible: model.isUser
                                            anchors.top: parent.top
                                            anchors.right: parent.right
                                            width: 16; height: 16
                                            color: parent.color
                                        }
                                        Rectangle {
                                            visible: !model.isUser
                                            anchors.top: parent.top
                                            anchors.left: parent.left
                                            width: 16; height: 16
                                            color: parent.color
                                        }
                                        /* Restore rounded for the opposite top corner */
                                        Rectangle {
                                            visible: model.isUser
                                            anchors.top: parent.top
                                            anchors.left: parent.left
                                            width: 32; height: 32
                                            radius: 16
                                            color: parent.color
                                        }
                                        Rectangle {
                                            visible: !model.isUser
                                            anchors.top: parent.top
                                            anchors.right: parent.right
                                            width: 32; height: 32
                                            radius: 16
                                            color: parent.color
                                        }

                                        Text {
                                            id: msgText
                                            anchors.fill: parent
                                            anchors.margins: 12
                                            text: model.text
                                            font.pixelSize: 13
                                            font.family: Theme.fontFamily
                                            color: model.isUser ? "#FFFFFF" : Theme.text
                                            wrapMode: Text.Wrap
                                            lineHeight: 1.4
                                        }
                                    }

                                    /* Timestamp */
                                    Text {
                                        text: model.time
                                        font.pixelSize: 9
                                        font.family: Theme.fontFamily
                                        color: Theme.textMuted
                                        anchors.left: model.isUser ? undefined : parent.left
                                        anchors.right: model.isUser ? parent.right : undefined
                                        anchors.leftMargin: model.isUser ? 0 : 4
                                        anchors.rightMargin: model.isUser ? 4 : 0
                                    }
                                }
                            }
                        }
                    }
                }

                /* ── Capability Cards (visible when no messages) ── */
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: messageModel.count === 0

                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 20

                        /* Centered welcome */
                        Column {
                            Layout.alignment: Qt.AlignHCenter
                            spacing: 6

                            Components.CanvasIcon {
                                anchors.horizontalCenter: parent.horizontalCenter
                                iconName: "robot"
                                iconSize: 40
                                iconColor: Theme.primary
                            }

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "NeuralOS AI Assistant"
                                font.pixelSize: Theme.fontSizeXL
                                font.bold: true
                                font.family: Theme.fontFamily
                                color: Theme.text
                            }

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "What can I help you with today?"
                                font.pixelSize: Theme.fontSizeNormal
                                font.family: Theme.fontFamily
                                color: Theme.textDim
                            }
                        }

                        /* Capability Grid */
                        GridLayout {
                            Layout.alignment: Qt.AlignHCenter
                            columns: 2
                            rowSpacing: 10
                            columnSpacing: 10

                            Repeater {
                                model: [
                                    { icon: "file",   label: "Summarize Text" },
                                    { icon: "code",   label: "Generate Code" },
                                    { icon: "info",   label: "Explain Concept" },
                                    { icon: "search", label: "Search Files" }
                                ]

                                Rectangle {
                                    width: 140; height: 80
                                    radius: Theme.radiusSmall
                                    color: capCardMa.containsMouse ? Theme.hoverBg : Theme.surfaceAlt

                                    Behavior on color { ColorAnimation { duration: Theme.animFast } }

                                    Column {
                                        anchors.centerIn: parent
                                        spacing: 8

                                        Components.CanvasIcon {
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            iconName: modelData.icon
                                            iconSize: 20
                                            iconColor: Theme.primary
                                        }

                                        Text {
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            text: modelData.label
                                            font.pixelSize: 11
                                            font.family: Theme.fontFamily
                                            color: Theme.text
                                        }
                                    }

                                    MouseArea {
                                        id: capCardMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            msgInput.text = modelData.label + ": "
                                            msgInput.forceActiveFocus()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /* ── Agent Workflow Panel (visible during thinking) ── */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: thinkingCol.height + 20
                    Layout.leftMargin: 14
                    Layout.rightMargin: 14
                    Layout.bottomMargin: 6
                    radius: Theme.radiusSmall
                    color: Theme.surfaceAlt
                    visible: isThinking

                    Column {
                        id: thinkingCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 10
                        spacing: 8

                        Repeater {
                            model: thinkingSteps

                            Row {
                                spacing: 8
                                height: 20

                                /* Check icon (green) or spinner */
                                Item {
                                    width: 14; height: 14
                                    anchors.verticalCenter: parent.verticalCenter

                                    Components.CanvasIcon {
                                        anchors.centerIn: parent
                                        iconName: "check"
                                        iconSize: 14
                                        iconColor: Theme.success
                                        visible: model.done
                                    }

                                    /* Spinner for active but not done */
                                    Rectangle {
                                        anchors.centerIn: parent
                                        width: 12; height: 12
                                        radius: 6
                                        color: "transparent"
                                        border.width: 2
                                        border.color: model.active && !model.done ? Theme.primary : Theme.textMuted
                                        visible: !model.done
                                        opacity: model.active ? 1.0 : 0.4

                                        /* Spinning dot indicator */
                                        Rectangle {
                                            width: 3; height: 3; radius: 1.5
                                            color: Theme.primary
                                            visible: model.active && !model.done
                                            x: 4.5; y: -0.5

                                            RotationAnimator on rotation {
                                                from: 0; to: 360
                                                duration: 800
                                                loops: Animation.Infinite
                                                running: model.active && !model.done
                                            }
                                            transformOrigin: Item.Bottom
                                        }
                                    }
                                }

                                Text {
                                    text: model.label
                                    font.pixelSize: 12
                                    font.family: Theme.fontFamily
                                    color: model.done ? Theme.success
                                         : model.active ? Theme.text
                                         : Theme.textMuted
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                        }
                    }
                }

                /* ── Input Area ── */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    Layout.leftMargin: 14
                    Layout.rightMargin: 14
                    Layout.bottomMargin: 12
                    radius: Theme.radiusSmall
                    color: Theme.surface
                    border.width: 1
                    border.color: msgInput.activeFocus ? Theme.primary : Theme.surfaceLight

                    Behavior on border.color { ColorAnimation { duration: Theme.animFast } }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 14
                        anchors.rightMargin: 8
                        spacing: 8

                        TextInput {
                            id: msgInput
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            verticalAlignment: TextInput.AlignVCenter
                            font.pixelSize: 13
                            font.family: Theme.fontFamily
                            color: Theme.text
                            clip: true
                            selectByMouse: true
                            selectionColor: Theme.primary

                            Keys.onReturnPressed: sendMessage(text)
                            Keys.onEnterPressed: sendMessage(text)

                            Text {
                                visible: !msgInput.text && !msgInput.activeFocus
                                text: "Ask anything..."
                                font.pixelSize: 13
                                font.family: Theme.fontFamily
                                color: Theme.textMuted
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }

                        /* Attachment button */
                        Rectangle {
                            width: 32; height: 32
                            radius: Theme.radiusTiny
                            color: attachMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "attachment"
                                iconSize: 16
                                iconColor: Theme.textDim
                            }

                            MouseArea {
                                id: attachMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                            }
                        }

                        /* Send button */
                        Rectangle {
                            width: 34; height: 34
                            radius: Theme.radiusTiny
                            color: sendBtnMa.containsMouse
                                ? Qt.darker(Theme.primary, 1.1)
                                : Theme.primary

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }

                            Components.CanvasIcon {
                                anchors.centerIn: parent
                                iconName: "send"
                                iconSize: 16
                                iconColor: "#FFFFFF"
                            }

                            MouseArea {
                                id: sendBtnMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: sendMessage(msgInput.text)
                            }
                        }
                    }
                }
            }
        }
    }
}
