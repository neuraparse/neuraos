import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: taskMgrApp
    anchors.fill: parent

    property int currentTab: 0
    property int selectedPid: -1
    property string processFilter: ""
    property int sortCol: 3
    property bool sortAsc: false

    property var processes: [
        { pid: 1,   name: "systemd",        status: "Running",  cpu: 0.1,  mem: 8.2 },
        { pid: 42,  name: "neural-daemon",   status: "Running",  cpu: 12.4, mem: 156.3 },
        { pid: 87,  name: "npu-driver",      status: "Running",  cpu: 5.7,  mem: 64.1 },
        { pid: 123, name: "neuralos-dashboard", status: "Running", cpu: 8.3, mem: 245.6 },
        { pid: 156, name: "ai-runtime",      status: "Running",  cpu: 22.1, mem: 512.0 },
        { pid: 201, name: "network-mgr",     status: "Running",  cpu: 1.2,  mem: 32.4 },
        { pid: 234, name: "firewalld",       status: "Running",  cpu: 0.4,  mem: 18.7 },
        { pid: 267, name: "sshd",            status: "Sleeping", cpu: 0.0,  mem: 12.1 },
        { pid: 312, name: "pkg-manager",     status: "Sleeping", cpu: 0.0,  mem: 8.4 },
        { pid: 345, name: "crond",           status: "Running",  cpu: 0.1,  mem: 4.2 },
        { pid: 378, name: "journald",        status: "Running",  cpu: 0.3,  mem: 16.8 },
        { pid: 412, name: "bluetoothd",      status: "Sleeping", cpu: 0.0,  mem: 6.1 },
        { pid: 445, name: "udevd",           status: "Running",  cpu: 0.2,  mem: 10.5 },
        { pid: 478, name: "dbus-daemon",     status: "Running",  cpu: 0.1,  mem: 5.8 },
        { pid: 511, name: "npie-engine",     status: "Running",  cpu: 15.6, mem: 384.2 }
    ]

    property var services: [
        { name: "neural-daemon",   desc: "Neural Processing Service",     status: "Active" },
        { name: "npu-driver",      desc: "NPU Hardware Driver",           status: "Active" },
        { name: "ai-runtime",      desc: "AI Inference Runtime",          status: "Active" },
        { name: "npie-engine",     desc: "NPIE Processing Engine",        status: "Active" },
        { name: "firewalld",       desc: "System Firewall",               status: "Active" },
        { name: "sshd",            desc: "SSH Server",                    status: "Active" },
        { name: "network-mgr",     desc: "Network Manager",              status: "Active" },
        { name: "bluetoothd",      desc: "Bluetooth Service",             status: "Inactive" },
        { name: "crond",           desc: "Task Scheduler",               status: "Active" },
        { name: "docker",          desc: "Container Runtime",             status: "Inactive" },
        { name: "nginx",           desc: "Web Server",                    status: "Inactive" },
        { name: "postgresql",      desc: "Database Server",               status: "Inactive" }
    ]

    property var cpuHistory: []
    property var memHistory: []

    Timer {
        interval: 1500; running: true; repeat: true
        onTriggered: {
            var p = processes.slice()
            for (var i = 0; i < p.length; i++) {
                p[i] = Object.assign({}, p[i])
                p[i].cpu = Math.max(0, p[i].cpu + (Math.random() - 0.5) * 3)
                p[i].cpu = Math.round(p[i].cpu * 10) / 10
            }
            processes = p

            var ch = cpuHistory.slice()
            ch.push(SystemInfo.cpuUsage)
            if (ch.length > 30) ch.shift()
            cpuHistory = ch

            var mh = memHistory.slice()
            var memPct = SystemInfo.memoryTotal > 0 ? SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
            mh.push(memPct)
            if (mh.length > 30) mh.shift()
            memHistory = mh
        }
    }

    function getFilteredProcesses() {
        var list = processes.slice()
        if (processFilter !== "") {
            list = list.filter(function(p) {
                return p.name.toLowerCase().indexOf(processFilter.toLowerCase()) !== -1 ||
                       p.pid.toString().indexOf(processFilter) !== -1
            })
        }
        list.sort(function(a, b) {
            var va, vb
            if (sortCol === 0) { va = a.pid; vb = b.pid }
            else if (sortCol === 1) { va = a.name; vb = b.name }
            else if (sortCol === 2) { va = a.status; vb = b.status }
            else if (sortCol === 3) { va = a.cpu; vb = b.cpu }
            else { va = a.mem; vb = b.mem }
            if (va < vb) return sortAsc ? -1 : 1
            if (va > vb) return sortAsc ? 1 : -1
            return 0
        })
        return list
    }

    function killProcess(pid) {
        var p = processes.filter(function(proc) { return proc.pid !== pid })
        processes = p
        selectedPid = -1
    }

    function totalCpu() {
        var t = 0; for (var i = 0; i < processes.length; i++) t += processes[i].cpu; return t.toFixed(1)
    }

    function totalMem() {
        var t = 0; for (var i = 0; i < processes.length; i++) t += processes[i].mem; return t.toFixed(0)
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* Tab Bar */
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    spacing: 0

                    Repeater {
                        model: [
                            { label: "Processes", ico: "list" },
                            { label: "Performance", ico: "dashboard" },
                            { label: "Services", ico: "gear" }
                        ]

                        Rectangle {
                            Layout.preferredWidth: tabLbl.implicitWidth + 36
                            Layout.fillHeight: true
                            color: tabBtnMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            RowLayout {
                                anchors.centerIn: parent
                                spacing: 6

                                Components.CanvasIcon {
                                    iconName: modelData.ico; iconSize: 13
                                    iconColor: currentTab === index ? Theme.primary : Theme.textDim
                                }

                                Text {
                                    id: tabLbl
                                    text: modelData.label
                                    font.pixelSize: 12
                                    font.weight: currentTab === index ? Font.DemiBold : Font.Normal
                                    color: currentTab === index ? Theme.text : Theme.textDim
                                }
                            }

                            Rectangle {
                                anchors.bottom: parent.bottom
                                anchors.left: parent.left; anchors.right: parent.right
                                height: 2; color: Theme.primary
                                visible: currentTab === index
                            }

                            MouseArea {
                                id: tabBtnMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: currentTab = index
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* ─── Tab 0: Processes ─── */
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                visible: currentTab === 0

                /* Toolbar */
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 44
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 12; anchors.rightMargin: 12
                        spacing: 8

                        Rectangle {
                            Layout.fillWidth: true; Layout.preferredHeight: 28
                            radius: 14; color: Theme.surfaceAlt

                            RowLayout {
                                anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 6
                                Components.CanvasIcon { iconName: "search"; iconSize: 12; iconColor: Theme.textDim }
                                TextInput {
                                    Layout.fillWidth: true
                                    font.pixelSize: 12; color: Theme.text; clip: true
                                    selectByMouse: true; selectionColor: Theme.primary
                                    onTextChanged: processFilter = text
                                    Text {
                                        visible: !parent.text && !parent.activeFocus
                                        text: "Filter processes..."
                                        color: Theme.textMuted; font.pixelSize: 12
                                    }
                                }
                            }
                        }

                        Rectangle {
                            width: endLbl.implicitWidth + 24; height: 28
                            radius: Theme.radiusSmall
                            color: selectedPid >= 0 ? (endMa.containsMouse ? "#C42B1C" : Theme.error) : Theme.surfaceLight
                            opacity: selectedPid >= 0 ? 1 : 0.5

                            Text {
                                id: endLbl; anchors.centerIn: parent
                                text: "End Process"; font.pixelSize: 11; font.weight: Font.DemiBold
                                color: selectedPid >= 0 ? "#FFFFFF" : Theme.textMuted
                            }

                            MouseArea {
                                id: endMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: selectedPid >= 0 ? Qt.PointingHandCursor : Qt.ArrowCursor
                                onClicked: if (selectedPid >= 0) killProcess(selectedPid)
                            }
                        }
                    }
                }

                /* Column Headers */
                Rectangle {
                    Layout.fillWidth: true; height: 28
                    color: Theme.surfaceAlt

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; spacing: 0

                        Repeater {
                            model: [
                                { label: "PID", w: 65, col: 0 },
                                { label: "Name", w: -1, col: 1 },
                                { label: "Status", w: 80, col: 2 },
                                { label: "CPU %", w: 70, col: 3 },
                                { label: "Memory", w: 90, col: 4 }
                            ]

                            Rectangle {
                                Layout.preferredWidth: modelData.w > 0 ? modelData.w : 0
                                Layout.fillWidth: modelData.w < 0
                                Layout.fillHeight: true
                                color: "transparent"

                                RowLayout {
                                    anchors.fill: parent; spacing: 4
                                    Text {
                                        text: modelData.label
                                        font.pixelSize: 10; font.bold: true; color: Theme.textDim
                                    }
                                    Text {
                                        visible: sortCol === modelData.col
                                        text: sortAsc ? "\u25B2" : "\u25BC"
                                        font.pixelSize: 8; color: Theme.primary
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (sortCol === modelData.col) sortAsc = !sortAsc
                                        else { sortCol = modelData.col; sortAsc = false }
                                    }
                                }
                            }
                        }
                    }
                }

                /* Process List */
                ListView {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    clip: true; model: getFilteredProcesses()

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded; width: 4
                        contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                    }

                    delegate: Rectangle {
                        width: ListView.view ? ListView.view.width : 0; height: 32
                        color: modelData.pid === selectedPid ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.12)
                             : procRowMa.containsMouse ? Theme.hoverBg : "transparent"

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; spacing: 0

                            Text { Layout.preferredWidth: 65; text: modelData.pid; font.pixelSize: 11; color: Theme.textDim }
                            Text { Layout.fillWidth: true; text: modelData.name; font.pixelSize: 11; color: Theme.text; font.weight: Font.DemiBold; elide: Text.ElideRight }
                            Rectangle {
                                Layout.preferredWidth: 80; Layout.preferredHeight: 18
                                color: "transparent"
                                Rectangle {
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: statusTxt.implicitWidth + 10; height: 16; radius: 8
                                    color: modelData.status === "Running" ? Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.12) : Qt.rgba(Theme.textMuted.r, Theme.textMuted.g, Theme.textMuted.b, 0.12)
                                    Text { id: statusTxt; anchors.centerIn: parent; text: modelData.status; font.pixelSize: 9; color: modelData.status === "Running" ? Theme.success : Theme.textMuted }
                                }
                            }
                            Text {
                                Layout.preferredWidth: 70
                                text: modelData.cpu.toFixed(1) + "%"
                                font.pixelSize: 11
                                color: modelData.cpu > 15 ? Theme.error : modelData.cpu > 5 ? Theme.warning : Theme.textDim
                            }
                            Text { Layout.preferredWidth: 90; text: modelData.mem.toFixed(1) + " MB"; font.pixelSize: 11; color: Theme.textDim }
                        }

                        MouseArea {
                            id: procRowMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: selectedPid = (selectedPid === modelData.pid ? -1 : modelData.pid)
                        }
                    }
                }
            }

            /* ─── Tab 1: Performance ─── */
            Flickable {
                Layout.fillWidth: true; Layout.fillHeight: true
                contentHeight: perfCol.height; clip: true
                visible: currentTab === 1

                ColumnLayout {
                    id: perfCol; width: parent.width; spacing: 12
                    anchors.margins: 12

                    Item { height: 12 }

                    /* Gauge Row */
                    RowLayout {
                        Layout.fillWidth: true; Layout.leftMargin: 12; Layout.rightMargin: 12; spacing: 12

                        Repeater {
                            model: [
                                { label: "CPU", val: SystemInfo.cpuUsage, clr: Theme.primary },
                                { label: "Memory", val: SystemInfo.memoryTotal > 0 ? SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0, clr: Theme.success },
                                { label: "Disk", val: SystemInfo.diskTotal > 0 ? SystemInfo.diskUsed / SystemInfo.diskTotal * 100 : 0, clr: Theme.warning },
                                { label: "NPU", val: NPUMonitor.utilizationPercent || 0, clr: Theme.secondary }
                            ]

                            Rectangle {
                                Layout.fillWidth: true; height: 140; radius: Theme.radius; color: Theme.surface
                                Components.CircularGauge {
                                    anchors.centerIn: parent; width: 100; height: 100
                                    value: modelData.val; gaugeColor: modelData.clr; label: modelData.label
                                }
                            }
                        }
                    }

                    /* CPU History Chart */
                    Rectangle {
                        Layout.fillWidth: true; Layout.leftMargin: 12; Layout.rightMargin: 12
                        height: 140; radius: Theme.radius; color: Theme.surface

                        Text {
                            anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 10
                            text: "CPU Usage (30s)"; color: Theme.textDim; font.pixelSize: 11; font.bold: true
                        }

                        Canvas {
                            id: cpuCanvas; anchors.fill: parent; anchors.topMargin: 28; anchors.margins: 10
                            property var data: cpuHistory
                            onDataChanged: requestPaint()
                            onPaint: drawChart(getContext("2d"), data, Theme.primary)
                        }
                    }

                    /* Memory History Chart */
                    Rectangle {
                        Layout.fillWidth: true; Layout.leftMargin: 12; Layout.rightMargin: 12
                        height: 140; radius: Theme.radius; color: Theme.surface

                        Text {
                            anchors.top: parent.top; anchors.left: parent.left; anchors.margins: 10
                            text: "Memory Usage (30s)"; color: Theme.textDim; font.pixelSize: 11; font.bold: true
                        }

                        Canvas {
                            id: memCanvas; anchors.fill: parent; anchors.topMargin: 28; anchors.margins: 10
                            property var data: memHistory
                            onDataChanged: requestPaint()
                            onPaint: drawChart(getContext("2d"), data, Theme.success)
                        }
                    }

                    /* System Info */
                    Rectangle {
                        Layout.fillWidth: true; Layout.leftMargin: 12; Layout.rightMargin: 12
                        height: infoCol.height + 24; radius: Theme.radius; color: Theme.surface

                        ColumnLayout {
                            id: infoCol; anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 12; spacing: 6

                            Text { text: "System Information"; font.pixelSize: 13; font.bold: true; color: Theme.text }
                            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                            Repeater {
                                model: [
                                    { k: "Hostname", v: SystemInfo.hostname },
                                    { k: "Kernel", v: SystemInfo.kernelVersion },
                                    { k: "Uptime", v: SystemInfo.uptime },
                                    { k: "CPU Temp", v: SystemInfo.cpuTemp.toFixed(1) + "\u00B0C" },
                                    { k: "Processes", v: processes.length + " total" },
                                    { k: "NPU Devices", v: NPUMonitor.deviceCount + " detected" }
                                ]

                                RowLayout {
                                    Layout.fillWidth: true
                                    Text { Layout.preferredWidth: 120; text: modelData.k; font.pixelSize: 11; color: Theme.textDim }
                                    Text { text: modelData.v; font.pixelSize: 11; color: Theme.text }
                                }
                            }
                        }
                    }

                    Item { height: 12 }
                }
            }

            /* ─── Tab 2: Services ─── */
            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                spacing: 0; visible: currentTab === 2

                /* Service Header */
                Rectangle {
                    Layout.fillWidth: true; height: 28; color: Theme.surfaceAlt

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12

                        Text { Layout.preferredWidth: 160; text: "Service"; font.pixelSize: 10; font.bold: true; color: Theme.textDim }
                        Text { Layout.fillWidth: true; text: "Description"; font.pixelSize: 10; font.bold: true; color: Theme.textDim }
                        Text { Layout.preferredWidth: 80; text: "Status"; font.pixelSize: 10; font.bold: true; color: Theme.textDim }
                    }
                }

                ListView {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    clip: true; model: services

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded; width: 4
                        contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                    }

                    delegate: Rectangle {
                        width: ListView.view ? ListView.view.width : 0; height: 36
                        color: svcMa.containsMouse ? Theme.hoverBg : "transparent"

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12

                            Text {
                                Layout.preferredWidth: 160; text: modelData.name
                                font.pixelSize: 11; font.weight: Font.DemiBold; color: Theme.text
                            }
                            Text {
                                Layout.fillWidth: true; text: modelData.desc
                                font.pixelSize: 11; color: Theme.textDim; elide: Text.ElideRight
                            }
                            Rectangle {
                                width: svcStatusLbl.implicitWidth + 14; height: 18; radius: 9
                                color: modelData.status === "Active"
                                    ? Qt.rgba(Theme.success.r, Theme.success.g, Theme.success.b, 0.12)
                                    : Qt.rgba(Theme.textMuted.r, Theme.textMuted.g, Theme.textMuted.b, 0.12)

                                Text {
                                    id: svcStatusLbl; anchors.centerIn: parent
                                    text: modelData.status; font.pixelSize: 9
                                    color: modelData.status === "Active" ? Theme.success : Theme.textMuted
                                }
                            }
                        }

                        MouseArea { id: svcMa; anchors.fill: parent; hoverEnabled: true }
                    }
                }
            }

            /* Status Bar */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 28
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; spacing: 16

                    Text { text: "Processes: " + processes.length; font.pixelSize: 10; color: Theme.textDim }
                    Text { text: "CPU: " + totalCpu() + "%"; font.pixelSize: 10; color: Theme.textDim }
                    Text { text: "Memory: " + totalMem() + " MB"; font.pixelSize: 10; color: Theme.textDim }
                    Item { Layout.fillWidth: true }
                    Text { text: "Uptime: " + SystemInfo.uptime; font.pixelSize: 10; color: Theme.textDim }
                }
            }
        }
    }

    function drawChart(ctx, data, lineColor) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        var w = ctx.canvas.width; var h = ctx.canvas.height

        ctx.strokeStyle = Theme.darkMode ? Qt.rgba(1, 1, 1, 0.05) : Qt.rgba(0, 0, 0, 0.06)
        ctx.lineWidth = 1
        for (var g = 0; g < 5; g++) {
            var gy = h * g / 4; ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke()
        }

        if (data.length < 2) return
        var step = w / 29
        ctx.beginPath()
        ctx.moveTo(0, h - (data[0] / 100 * h))
        for (var i = 1; i < data.length; i++) ctx.lineTo(i * step, h - (data[i] / 100 * h))
        ctx.strokeStyle = lineColor; ctx.lineWidth = 2; ctx.stroke()

        ctx.lineTo((data.length - 1) * step, h); ctx.lineTo(0, h); ctx.closePath()
        ctx.fillStyle = Qt.rgba(lineColor.r, lineColor.g, lineColor.b, 0.08); ctx.fill()
    }
}
