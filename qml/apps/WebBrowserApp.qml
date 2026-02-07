import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: browserApp
    anchors.fill: parent

    property var tabs: ListModel {}
    property int activeTab: 0
    property bool isLoading: false

    /* ── Helper functions ── */

    function addTab() {
        tabs.append({ title: "New Tab", url: "neuralos://newtab", content: "newtab" })
        activeTab = tabs.count - 1
        urlField.text = "neuralos://newtab"
    }

    function closeTab(index) {
        if (tabs.count <= 1) return
        tabs.remove(index)
        if (activeTab >= tabs.count) activeTab = tabs.count - 1
        else if (activeTab > index) activeTab--
        urlField.text = tabs.get(activeTab).url
    }

    function navigateUrl(url) {
        if (tabs.count === 0) return
        var contentType = "error"
        if (url === "neuralos://newtab" || url === "about:newtab")
            contentType = "newtab"
        else if (url.indexOf("docs.neuralos") !== -1)
            contentType = "docs"
        else if (url.indexOf("neuralos.dev") !== -1 && url.indexOf("docs") === -1)
            contentType = "newtab"

        var titleStr = url.replace(/^https?:\/\//, "").replace(/^neuralos:\/\//, "").split("/")[0]
        if (contentType === "newtab") titleStr = "New Tab"
        else if (contentType === "docs") titleStr = "NeuralOS Docs"

        tabs.set(activeTab, { title: titleStr, url: url, content: contentType })
        isLoading = true
        loadingTimer.restart()
    }

    function switchTab(index) {
        if (index < 0 || index >= tabs.count) return
        activeTab = index
        urlField.text = tabs.get(activeTab).url
    }

    Timer {
        id: loadingTimer
        interval: 800
        onTriggered: isLoading = false
    }

    Component.onCompleted: {
        tabs.append({ title: "New Tab",        url: "neuralos://newtab",        content: "newtab" })
        tabs.append({ title: "NeuralOS Docs",  url: "https://docs.neuralos.dev", content: "docs"   })
        urlField.text = tabs.get(0).url
    }

    /* ── Main UI ── */

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ════════════ Tab Bar ════════════ */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 36
                color: Theme.surfaceAlt

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 6
                    anchors.rightMargin: 6
                    anchors.topMargin: 4
                    spacing: 2

                    Repeater {
                        model: tabs

                        Rectangle {
                            id: tabRect
                            property bool hovered: tabMouseArea.containsMouse

                            Layout.preferredWidth: Math.min(220,
                                Math.max(120, (browserApp.width - 50) / Math.max(1, tabs.count)))
                            Layout.preferredHeight: 32
                            Layout.alignment: Qt.AlignBottom

                            radius: Theme.radiusSmall
                            color: index === activeTab ? Theme.surface : hovered ? Theme.surfaceLight : "transparent"

                            /* Rounded top corners, flat bottom: clip a child rect */
                            Rectangle {
                                anchors.bottom: parent.bottom
                                anchors.left: parent.left
                                anchors.right: parent.right
                                height: Theme.radiusSmall
                                color: parent.color
                                visible: index === activeTab
                            }

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 6
                                spacing: 6

                                Components.CanvasIcon {
                                    iconName: "globe"
                                    iconSize: 12
                                    iconColor: index === activeTab ? Theme.primary : Theme.textDim
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: model.title
                                    font.pixelSize: 11
                                    font.weight: index === activeTab ? Font.DemiBold : Font.Normal
                                    font.family: Theme.fontFamily
                                    color: index === activeTab ? Theme.text : Theme.textDim
                                    elide: Text.ElideRight
                                }

                                /* Close button */
                                Rectangle {
                                    width: 18; height: 18; radius: 9
                                    color: tabCloseMa.containsMouse ? Theme.surfaceLight : "transparent"
                                    visible: tabRect.hovered || index === activeTab
                                    opacity: tabCloseMa.containsMouse ? 1.0 : 0.6

                                    Components.CanvasIcon {
                                        anchors.centerIn: parent
                                        iconName: "close"
                                        iconSize: 10
                                        iconColor: Theme.textDim
                                    }

                                    MouseArea {
                                        id: tabCloseMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: closeTab(index)
                                    }
                                }
                            }

                            MouseArea {
                                id: tabMouseArea
                                anchors.fill: parent
                                z: -1
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: switchTab(index)
                            }
                        }
                    }

                    /* New tab (+) button */
                    Rectangle {
                        Layout.preferredWidth: 32
                        Layout.preferredHeight: 32
                        Layout.alignment: Qt.AlignBottom
                        radius: Theme.radiusSmall
                        color: newTabMa.containsMouse ? Theme.surfaceLight : "transparent"

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "plus"
                            iconSize: 14
                            iconColor: Theme.textDim
                        }

                        MouseArea {
                            id: newTabMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: addTab()
                        }
                    }

                    Item { Layout.fillWidth: true }
                }
            }

            /* ════════════ Navigation + URL Bar ════════════ */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 44
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 4

                    /* Back */
                    Rectangle {
                        Layout.preferredWidth: 32; Layout.preferredHeight: 32
                        radius: Theme.radiusSmall
                        color: backMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        opacity: 0.5

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "arrow-left"
                            iconSize: 16
                            iconColor: Theme.textDim
                        }
                        MouseArea {
                            id: backMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        }
                    }

                    /* Forward */
                    Rectangle {
                        Layout.preferredWidth: 32; Layout.preferredHeight: 32
                        radius: Theme.radiusSmall
                        color: fwdMa.containsMouse ? Theme.surfaceAlt : "transparent"
                        opacity: 0.5

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "arrow-right"
                            iconSize: 16
                            iconColor: Theme.textDim
                        }
                        MouseArea {
                            id: fwdMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        }
                    }

                    /* Refresh */
                    Rectangle {
                        Layout.preferredWidth: 32; Layout.preferredHeight: 32
                        radius: Theme.radiusSmall
                        color: refreshMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "refresh"
                            iconSize: 16
                            iconColor: Theme.textDim
                        }
                        MouseArea {
                            id: refreshMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                isLoading = true
                                loadingTimer.restart()
                            }
                        }
                    }

                    /* URL Bar */
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 32
                        radius: 16
                        color: Theme.surfaceAlt
                        border.width: urlField.activeFocus ? 1 : 0
                        border.color: Theme.primary
                        Behavior on border.width { NumberAnimation { duration: 100 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            spacing: 6

                            Components.CanvasIcon {
                                iconName: "lock"
                                iconSize: 12
                                iconColor: Theme.success
                            }

                            TextInput {
                                id: urlField
                                Layout.fillWidth: true
                                font.pixelSize: 12
                                font.family: Theme.fontFamily
                                color: Theme.text
                                clip: true
                                selectByMouse: true
                                selectionColor: Theme.primary
                                selectedTextColor: "#FFFFFF"
                                verticalAlignment: TextInput.AlignVCenter

                                onAccepted: navigateUrl(text)
                            }
                        }
                    }

                    /* Bookmark */
                    Rectangle {
                        Layout.preferredWidth: 32; Layout.preferredHeight: 32
                        radius: Theme.radiusSmall
                        color: bookmarkMa.containsMouse ? Theme.surfaceAlt : "transparent"

                        Components.CanvasIcon {
                            anchors.centerIn: parent
                            iconName: "bookmark"
                            iconSize: 16
                            iconColor: Theme.textDim
                        }
                        MouseArea {
                            id: bookmarkMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        }
                    }
                }
            }

            /* Separator */
            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: Theme.surfaceLight
            }

            /* ════════════ Content Area ════════════ */
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true

                /* Loading bar at very top */
                Components.ProgressBar {
                    id: loadBar
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 2
                    z: 10
                    indeterminate: true
                    barColor: Theme.primary
                    visible: isLoading
                }

                StackLayout {
                    anchors.fill: parent
                    currentIndex: {
                        if (tabs.count === 0) return 0
                        var c = tabs.get(activeTab).content
                        if (c === "newtab") return 0
                        if (c === "docs")   return 1
                        return 2
                    }

                    /* ──────── Page 0: New Tab ──────── */
                    Flickable {
                        contentHeight: newTabContent.height + 60
                        clip: true
                        flickableDirection: Flickable.VerticalFlick

                        Item {
                            id: newTabContent
                            width: parent.width
                            height: newTabCol.height + 80

                            ColumnLayout {
                                id: newTabCol
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: 80
                                width: Math.min(560, parent.width - 40)
                                spacing: 32

                                /* NeuralOS Logo Text */
                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "NeuralOS"
                                    font.pixelSize: Theme.fontSizeHuge
                                    font.weight: Font.Bold
                                    font.family: Theme.fontFamily
                                    color: Theme.primary
                                }

                                /* Search Bar */
                                Rectangle {
                                    Layout.alignment: Qt.AlignHCenter
                                    Layout.preferredWidth: 500
                                    Layout.preferredHeight: 44
                                    radius: 22
                                    color: Theme.surfaceAlt
                                    border.width: searchInput.activeFocus ? 1 : 0
                                    border.color: Theme.primary

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 16
                                        anchors.rightMargin: 16
                                        spacing: 10

                                        Components.CanvasIcon {
                                            iconName: "search"
                                            iconSize: 16
                                            iconColor: Theme.textDim
                                        }

                                        TextInput {
                                            id: searchInput
                                            Layout.fillWidth: true
                                            font.pixelSize: 14
                                            font.family: Theme.fontFamily
                                            color: Theme.text
                                            clip: true
                                            selectByMouse: true
                                            selectionColor: Theme.primary
                                            selectedTextColor: "#FFFFFF"
                                            verticalAlignment: TextInput.AlignVCenter

                                            Text {
                                                anchors.verticalCenter: parent.verticalCenter
                                                text: "Search the web..."
                                                font.pixelSize: 14
                                                font.family: Theme.fontFamily
                                                color: Theme.textMuted
                                                visible: !searchInput.text && !searchInput.activeFocus
                                            }

                                            onAccepted: {
                                                if (text.length > 0) {
                                                    var searchUrl = "https://search.neuralos.dev/?q=" + text
                                                    urlField.text = searchUrl
                                                    navigateUrl(searchUrl)
                                                }
                                            }
                                        }
                                    }
                                }

                                /* Quick Links Label */
                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    Layout.topMargin: 8
                                    text: "Quick Links"
                                    font.pixelSize: Theme.fontSizeSmall
                                    font.family: Theme.fontFamily
                                    color: Theme.textMuted
                                }

                                /* Quick Links Grid (2 rows x 4 columns) */
                                GridLayout {
                                    Layout.alignment: Qt.AlignHCenter
                                    columns: 4
                                    rowSpacing: 16
                                    columnSpacing: 24

                                    Repeater {
                                        model: [
                                            { name: "NeuralOS Docs", icon: "file",     clr: "#5B9AFF", linkUrl: "https://docs.neuralos.dev",  linkContent: "docs"   },
                                            { name: "GitHub",        icon: "code",     clr: "#8B8FA2", linkUrl: "https://github.com",         linkContent: "error"  },
                                            { name: "Wikipedia",     icon: "globe",    clr: "#34D399", linkUrl: "https://wikipedia.org",      linkContent: "error"  },
                                            { name: "YouTube",       icon: "play",     clr: "#F87171", linkUrl: "https://youtube.com",        linkContent: "error"  },
                                            { name: "Reddit",        icon: "chat",     clr: "#FBBF24", linkUrl: "https://reddit.com",         linkContent: "error"  },
                                            { name: "Stack Overflow",icon: "database", clr: "#F59E0B", linkUrl: "https://stackoverflow.com",  linkContent: "error"  },
                                            { name: "MDN Web Docs",  icon: "bookmark", clr: "#A78BFA", linkUrl: "https://developer.mozilla.org", linkContent: "error"},
                                            { name: "Anthropic",     icon: "neural",   clr: "#EC4899", linkUrl: "https://anthropic.com",      linkContent: "error"  }
                                        ]

                                        Column {
                                            spacing: 8
                                            width: 72

                                            Rectangle {
                                                width: 40; height: 40
                                                radius: 20
                                                color: qlMa.containsMouse ? Theme.surfaceLight : Theme.surfaceAlt
                                                anchors.horizontalCenter: parent.horizontalCenter
                                                Behavior on color { ColorAnimation { duration: 120 } }

                                                Components.CanvasIcon {
                                                    anchors.centerIn: parent
                                                    iconName: modelData.icon
                                                    iconSize: 18
                                                    iconColor: modelData.clr
                                                }
                                            }

                                            Text {
                                                width: parent.width
                                                text: modelData.name
                                                font.pixelSize: 10
                                                font.family: Theme.fontFamily
                                                color: Theme.textDim
                                                horizontalAlignment: Text.AlignHCenter
                                                elide: Text.ElideRight
                                            }

                                            MouseArea {
                                                id: qlMa
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: {
                                                    tabs.set(activeTab, {
                                                        title: modelData.name,
                                                        url: modelData.linkUrl,
                                                        content: modelData.linkContent
                                                    })
                                                    urlField.text = modelData.linkUrl
                                                    isLoading = true
                                                    loadingTimer.restart()
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* ──────── Page 1: NeuralOS Docs ──────── */
                    Flickable {
                        contentHeight: docsContent.height + 40
                        clip: true
                        flickableDirection: Flickable.VerticalFlick

                        RowLayout {
                            id: docsContent
                            width: parent.width
                            spacing: 0

                            /* Sidebar */
                            Rectangle {
                                Layout.preferredWidth: 220
                                Layout.fillHeight: true
                                Layout.minimumHeight: 600
                                color: Theme.surface

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 16
                                    spacing: 4

                                    Text {
                                        text: "Documentation"
                                        font.pixelSize: Theme.fontSizeLarge
                                        font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true; height: 1
                                        Layout.topMargin: 8; Layout.bottomMargin: 8
                                        color: Theme.surfaceLight
                                    }

                                    Repeater {
                                        model: [
                                            { section: "Getting Started",   active: true  },
                                            { section: "Architecture",      active: false },
                                            { section: "NPU Programming",   active: false },
                                            { section: "AI Runtime API",    active: false },
                                            { section: "System Services",   active: false },
                                            { section: "Security Model",    active: false },
                                            { section: "Package Management",active: false },
                                            { section: "Contributing",      active: false }
                                        ]

                                        Rectangle {
                                            Layout.fillWidth: true
                                            height: 32
                                            radius: Theme.radiusTiny
                                            color: modelData.active ? Theme.primary
                                                 : docSidebarMa.containsMouse ? Theme.surfaceAlt
                                                 : "transparent"

                                            Text {
                                                anchors.verticalCenter: parent.verticalCenter
                                                anchors.left: parent.left
                                                anchors.leftMargin: 12
                                                text: modelData.section
                                                font.pixelSize: 12
                                                font.weight: modelData.active ? Font.DemiBold : Font.Normal
                                                font.family: Theme.fontFamily
                                                color: modelData.active ? "#FFFFFF" : Theme.textDim
                                            }

                                            MouseArea {
                                                id: docSidebarMa
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                            }
                                        }
                                    }

                                    Item { Layout.fillHeight: true }
                                }
                            }

                            Rectangle {
                                Layout.preferredWidth: 1
                                Layout.fillHeight: true
                                Layout.minimumHeight: 600
                                color: Theme.surfaceLight
                            }

                            /* Main content */
                            Flickable {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.minimumHeight: 600
                                contentHeight: docsMainCol.height + 60
                                clip: true
                                flickableDirection: Flickable.VerticalFlick

                                ColumnLayout {
                                    id: docsMainCol
                                    width: parent.width
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.margins: 32
                                    spacing: 16

                                    Item { Layout.preferredHeight: 8 }

                                    /* Breadcrumb */
                                    RowLayout {
                                        spacing: 6
                                        Text { text: "Docs"; font.pixelSize: 12; color: Theme.primary; font.family: Theme.fontFamily }
                                        Text { text: "/"; font.pixelSize: 12; color: Theme.textMuted; font.family: Theme.fontFamily }
                                        Text { text: "Getting Started"; font.pixelSize: 12; color: Theme.textDim; font.family: Theme.fontFamily }
                                    }

                                    /* Title */
                                    Text {
                                        text: "Getting Started with NeuralOS"
                                        font.pixelSize: 28
                                        font.weight: Font.Bold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "Welcome to NeuralOS, the AI-native operating system designed for the next generation of computing. This guide will help you understand the core concepts and get your environment configured."
                                        font.pixelSize: 14
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true; height: 1
                                        Layout.topMargin: 8; Layout.bottomMargin: 8
                                        color: Theme.surfaceLight
                                    }

                                    /* Section: System Requirements */
                                    Text {
                                        text: "System Requirements"
                                        font.pixelSize: 20
                                        font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "NeuralOS requires a 64-bit processor with NPU (Neural Processing Unit) support. Minimum 16 GB RAM is recommended for full AI runtime capabilities. A compatible GPU with Vulkan 1.3+ support enables hardware-accelerated rendering and inference pipelines."
                                        font.pixelSize: 13
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    /* Code block */
                                    Rectangle {
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: codeBlockText.height + 24
                                        radius: Theme.radiusTiny
                                        color: Theme.surfaceAlt
                                        border.width: 1
                                        border.color: Theme.surfaceLight

                                        Text {
                                            id: codeBlockText
                                            anchors.left: parent.left
                                            anchors.right: parent.right
                                            anchors.verticalCenter: parent.verticalCenter
                                            anchors.margins: 14
                                            text: "$ neuralos --version\nNeuralOS v4.0.0 (kernel 6.12-npu)\nNPU Runtime: active (8 TOPS)\nAI Subsystem: ready"
                                            font.pixelSize: 12
                                            font.family: "Courier New"
                                            color: Theme.success
                                            wrapMode: Text.WordWrap
                                            lineHeight: 1.5
                                        }
                                    }

                                    /* Section: Architecture */
                                    Text {
                                        Layout.topMargin: 8
                                        text: "Architecture Overview"
                                        font.pixelSize: 20
                                        font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "NeuralOS is built on a microkernel architecture with a dedicated AI runtime layer. The NPU scheduler manages neural workloads alongside traditional CPU tasks, enabling real-time inference without impacting system responsiveness."
                                        font.pixelSize: 13
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "The AI Runtime API provides a unified interface for deploying models across CPU, GPU, and NPU backends. Models are automatically optimized for the available hardware using the NeuralOS compilation pipeline."
                                        font.pixelSize: 13
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    /* Section: NPU Programming */
                                    Text {
                                        Layout.topMargin: 8
                                        text: "NPU Programming Model"
                                        font.pixelSize: 20
                                        font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "The Neural Processing Unit exposes a stream-based programming model. Developers submit inference graphs to the NPU scheduler, which handles memory allocation, operator fusion, and execution ordering automatically."
                                        font.pixelSize: 13
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: codeBlock2Text.height + 24
                                        radius: Theme.radiusTiny
                                        color: Theme.surfaceAlt
                                        border.width: 1
                                        border.color: Theme.surfaceLight

                                        Text {
                                            id: codeBlock2Text
                                            anchors.left: parent.left
                                            anchors.right: parent.right
                                            anchors.verticalCenter: parent.verticalCenter
                                            anchors.margins: 14
                                            text: "import neuralos.npu as npu\n\ndevice = npu.get_device()\nmodel = npu.load_model(\"resnet50.npu\")\nresult = device.infer(model, input_tensor)\nprint(f\"Inference time: {result.latency_ms:.1f}ms\")"
                                            font.pixelSize: 12
                                            font.family: "Courier New"
                                            color: Theme.accent
                                            wrapMode: Text.WordWrap
                                            lineHeight: 1.5
                                        }
                                    }

                                    /* Section: AI Runtime API */
                                    Text {
                                        Layout.topMargin: 8
                                        text: "AI Runtime API"
                                        font.pixelSize: 20
                                        font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily
                                        color: Theme.text
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: "The AI Runtime provides system-level services for model management, inference scheduling, and resource allocation. Applications can request AI capabilities through the standard D-Bus interface or the native C++/Python bindings."
                                        font.pixelSize: 13
                                        font.family: Theme.fontFamily
                                        color: Theme.textDim
                                        wrapMode: Text.WordWrap
                                        lineHeight: 1.5
                                    }

                                    /* Info card */
                                    Rectangle {
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: infoCardCol.height + 24
                                        radius: Theme.radiusSmall
                                        color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.08)
                                        border.width: 1
                                        border.color: Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)

                                        RowLayout {
                                            id: infoCardCol
                                            anchors.left: parent.left
                                            anchors.right: parent.right
                                            anchors.verticalCenter: parent.verticalCenter
                                            anchors.margins: 14
                                            spacing: 10

                                            Components.CanvasIcon {
                                                iconName: "info"
                                                iconSize: 16
                                                iconColor: Theme.primary
                                            }

                                            Text {
                                                Layout.fillWidth: true
                                                text: "NeuralOS v4.0 introduces the Autonomous Agent Framework, allowing AI agents to operate within sandboxed environments with controlled system access."
                                                font.pixelSize: 13
                                                font.family: Theme.fontFamily
                                                color: Theme.text
                                                wrapMode: Text.WordWrap
                                                lineHeight: 1.4
                                            }
                                        }
                                    }

                                    Item { Layout.preferredHeight: 40 }
                                }
                            }
                        }
                    }

                    /* ──────── Page 2: Error / 404 ──────── */
                    Item {
                        ColumnLayout {
                            anchors.centerIn: parent
                            spacing: 16

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "404"
                                font.pixelSize: 72
                                font.weight: Font.Bold
                                font.family: Theme.fontFamily
                                color: Theme.textMuted
                            }

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "Page not found"
                                font.pixelSize: Theme.fontSizeXL
                                font.family: Theme.fontFamily
                                color: Theme.textDim
                            }

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: tabs.count > 0 ? tabs.get(activeTab).url : ""
                                font.pixelSize: Theme.fontSizeSmall
                                font.family: Theme.fontFamily
                                color: Theme.textMuted
                            }

                            Item { Layout.preferredHeight: 8 }

                            Rectangle {
                                Layout.alignment: Qt.AlignHCenter
                                width: 160; height: 38
                                radius: 19
                                color: goHomeMa.containsMouse ? Theme.primaryDim : Theme.primary
                                Behavior on color { ColorAnimation { duration: 120 } }

                                Text {
                                    anchors.centerIn: parent
                                    text: "Go to New Tab"
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily
                                    color: "#FFFFFF"
                                }

                                MouseArea {
                                    id: goHomeMa
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        tabs.set(activeTab, { title: "New Tab", url: "neuralos://newtab", content: "newtab" })
                                        urlField.text = "neuralos://newtab"
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
