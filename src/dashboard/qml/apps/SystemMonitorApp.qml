import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: sysMonApp
    anchors.fill: parent

    // ── Current tab ──────────────────────────────────────────────
    property int currentTab: 0

    // ── Live gauge values ────────────────────────────────────────
    property real cpuVal:  SystemInfo.cpuUsage
    property real memVal:  SystemInfo.memoryTotal > 0
                           ? SystemInfo.memoryUsed / SystemInfo.memoryTotal * 100 : 0
    property real diskVal: SystemInfo.diskTotal > 0
                           ? SystemInfo.diskUsed / SystemInfo.diskTotal * 100 : 0
    property real npuVal:  typeof NPUMonitor !== "undefined" && NPUMonitor.npuUtilization !== undefined ? NPUMonitor.npuUtilization : 34

    // ── Performance history (60 data points) ─────────────────────
    property var cpuHistory: []
    property var memHistory: []

    Component.onCompleted: {
        var c = [], m = [];
        for (var i = 0; i < 60; i++) {
            c.push(20 + Math.random() * 30);
            m.push(35 + Math.random() * 15);
        }
        cpuHistory = c;
        memHistory = m;
    }

    // ── Periodic update timer ────────────────────────────────────
    Timer {
        id: updateTimer
        interval: 1000; running: true; repeat: true
        onTriggered: {
            sysMonApp.cpuVal  = Math.max(2,  Math.min(100, sysMonApp.cpuVal  + (Math.random() - 0.48) * 8));
            sysMonApp.memVal  = Math.max(10, Math.min(95,  sysMonApp.memVal  + (Math.random() - 0.5)  * 3));
            sysMonApp.npuVal  = Math.max(0,  Math.min(100, sysMonApp.npuVal  + (Math.random() - 0.5)  * 6));

            var c = cpuHistory.slice(); c.push(sysMonApp.cpuVal);
            if (c.length > 60) c.shift();
            cpuHistory = c;

            var m = memHistory.slice(); m.push(sysMonApp.memVal);
            if (m.length > 60) m.shift();
            memHistory = m;

            cpuGaugeCanvas.requestPaint();
            memGaugeCanvas.requestPaint();
            diskGaugeCanvas.requestPaint();
            npuGaugeCanvas.requestPaint();
            cpuGraphCanvas.requestPaint();
            memGraphCanvas.requestPaint();
        }
    }

    // ── Mock process list ────────────────────────────────────────
    property var processList: [
        { pid: 1,    name: "systemd",            cpu: "0.1",  mem: "12 MB",   status: "Running"  },
        { pid: 87,   name: "kthreadd",           cpu: "0.0",  mem: "0 MB",    status: "Sleeping" },
        { pid: 142,  name: "udevd",              cpu: "0.2",  mem: "8 MB",    status: "Running"  },
        { pid: 310,  name: "Xorg",               cpu: "3.4",  mem: "145 MB",  status: "Running"  },
        { pid: 455,  name: "neuraos-dashboard",   cpu: "5.2",  mem: "238 MB",  status: "Running"  },
        { pid: 512,  name: "neuraos-compositor",  cpu: "2.8",  mem: "96 MB",   status: "Running"  },
        { pid: 620,  name: "npie-daemon",         cpu: "1.7",  mem: "64 MB",   status: "Running"  },
        { pid: 733,  name: "python3",             cpu: "4.1",  mem: "182 MB",  status: "Running"  },
        { pid: 801,  name: "node",                cpu: "2.3",  mem: "120 MB",  status: "Running"  },
        { pid: 845,  name: "bash",                cpu: "0.0",  mem: "5 MB",    status: "Sleeping" },
        { pid: 892,  name: "dbus-daemon",         cpu: "0.1",  mem: "4 MB",    status: "Sleeping" },
        { pid: 940,  name: "pulseaudio",          cpu: "0.5",  mem: "18 MB",   status: "Sleeping" },
        { pid: 1021, name: "networkmanager",      cpu: "0.3",  mem: "22 MB",   status: "Running"  },
        { pid: 1100, name: "sshd",                cpu: "0.0",  mem: "6 MB",    status: "Sleeping" },
        { pid: 1187, name: "rsyslogd",            cpu: "0.1",  mem: "10 MB",   status: "Running"  }
    ]

    // ── Helper: draw a circular arc gauge ────────────────────────
    function drawGauge(ctx, w, h, value, colorStr) {
        var cx = w / 2;
        var cy = h / 2;
        var r  = Math.min(cx, cy) - 8;
        var startAngle = Math.PI * 0.75;
        var sweep      = Math.PI * 1.5;
        var valAngle   = startAngle + sweep * Math.min(value, 100) / 100;

        ctx.clearRect(0, 0, w, h);

        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle, startAngle + sweep);
        ctx.lineWidth = 7;
        ctx.strokeStyle = String(Theme.surfaceLight);
        ctx.lineCap = "round";
        ctx.stroke();

        if (value > 0) {
            ctx.beginPath();
            ctx.arc(cx, cy, r, startAngle, valAngle);
            ctx.lineWidth = 7;
            ctx.strokeStyle = colorStr;
            ctx.lineCap = "round";
            ctx.stroke();
        }
    }

    // ── Helper: draw a line graph with gradient fill ─────────────
    function drawGraph(ctx, w, h, data, colorStr) {
        ctx.clearRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.06);
        ctx.lineWidth = 1;
        for (var g = 0; g <= 4; g++) {
            var gy = h * g / 4;
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(w, gy);
            ctx.stroke();
        }

        // Axis labels
        ctx.fillStyle = String(Theme.textMuted);
        ctx.font = "10px " + Theme.fontFamily;
        ctx.textAlign = "left";
        ctx.fillText("100%", 2, 12);
        ctx.fillText("50%",  2, h / 2 + 4);
        ctx.fillText("0%",   2, h - 2);

        if (data.length < 2) return;

        var padL = 30;
        var stepX = (w - padL) / 59;

        // Line path
        ctx.beginPath();
        ctx.moveTo(padL, h - (data[0] / 100 * h));
        for (var i = 1; i < data.length; i++) {
            ctx.lineTo(padL + i * stepX, h - (data[i] / 100 * h));
        }
        ctx.strokeStyle = colorStr;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Gradient fill below
        var lastX = padL + (data.length - 1) * stepX;
        ctx.lineTo(lastX, h);
        ctx.lineTo(padL, h);
        ctx.closePath();

        // Parse color string to RGBA components safely
        var r = 0.36, gv = 0.6, b = 1.0;
        if (colorStr === "#5B9AFF")      { r = 0.357; gv = 0.604; b = 1.0;   }
        else if (colorStr === "#A78BFA") { r = 0.655; gv = 0.545; b = 0.980; }

        var grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, Qt.rgba(r, gv, b, 0.25));
        grad.addColorStop(1, Qt.rgba(r, gv, b, 0.02));
        ctx.fillStyle = grad;
        ctx.fill();
    }

    // ── Main layout ──────────────────────────────────────────────
    Rectangle {
        anchors.fill: parent
        color: Theme.background

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // ── Tab bar ──────────────────────────────────────
            Rectangle {
                id: sysTabBar
                Layout.fillWidth: true
                Layout.preferredHeight: 44
                color: Theme.surface

                Row {
                    x: 16; y: (sysTabBar.height - height) / 2
                    spacing: 4

                    Repeater {
                        model: ["Overview", "Processes", "Performance"]
                        delegate: Rectangle {
                            width: 110; height: 32
                            radius: Theme.radiusSmall
                            color: sysMonApp.currentTab === index
                                   ? Theme.primary
                                   : tabMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                color: sysMonApp.currentTab === index ? "#FFFFFF" : Theme.textDim
                                font.pixelSize: 12
                                font.bold: sysMonApp.currentTab === index
                                font.family: Theme.fontFamily
                            }
                            MouseArea {
                                id: tabMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: sysMonApp.currentTab = index
                            }
                        }
                    }
                }

                Rectangle {
                    y: sysTabBar.height - 1
                    width: sysTabBar.width; height: 1
                    color: Theme.surfaceLight
                }
            }

            // ── Stacked pages ────────────────────────────────
            StackLayout {
                id: tabStack
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: sysMonApp.currentTab

                // ========== OVERVIEW TAB ==========
                Flickable {
                    contentHeight: overviewCol.height + 24
                    clip: true

                    ColumnLayout {
                        id: overviewCol
                        width: parent.width - 24
                        x: 12; y: 12
                        spacing: 14

                        // ── Gauge row (4 explicit cards, no Repeater) ──
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            // CPU gauge card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 190
                                radius: Theme.radiusSmall
                                color: Theme.surface

                                Column {
                                    anchors.centerIn: parent
                                    spacing: 4

                                    Canvas {
                                        id: cpuGaugeCanvas
                                        width: 110; height: 110
                                        onPaint: drawGauge(getContext("2d"), width, height, sysMonApp.cpuVal, "#5B9AFF")
                                    }
                                    Text {
                                        text: Math.round(sysMonApp.cpuVal) + "%"
                                        color: "#5B9AFF"
                                        font.pixelSize: 22; font.bold: true; font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "CPU"
                                        color: Theme.textDim
                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                    }
                                }
                            }

                            // Memory gauge card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 190
                                radius: Theme.radiusSmall
                                color: Theme.surface

                                Column {
                                    anchors.centerIn: parent
                                    spacing: 4

                                    Canvas {
                                        id: memGaugeCanvas
                                        width: 110; height: 110
                                        onPaint: drawGauge(getContext("2d"), width, height, sysMonApp.memVal, "#A78BFA")
                                    }
                                    Text {
                                        text: Math.round(sysMonApp.memVal) + "%"
                                        color: "#A78BFA"
                                        font.pixelSize: 22; font.bold: true; font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "Memory"
                                        color: Theme.textDim
                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                    }
                                }
                            }

                            // Disk gauge card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 190
                                radius: Theme.radiusSmall
                                color: Theme.surface

                                Column {
                                    anchors.centerIn: parent
                                    spacing: 4

                                    Canvas {
                                        id: diskGaugeCanvas
                                        width: 110; height: 110
                                        onPaint: drawGauge(getContext("2d"), width, height, sysMonApp.diskVal, "#38BDF8")
                                    }
                                    Text {
                                        text: Math.round(sysMonApp.diskVal) + "%"
                                        color: "#38BDF8"
                                        font.pixelSize: 22; font.bold: true; font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "Disk"
                                        color: Theme.textDim
                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                    }
                                }
                            }

                            // NPU gauge card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 190
                                radius: Theme.radiusSmall
                                color: Theme.surface

                                Column {
                                    anchors.centerIn: parent
                                    spacing: 4

                                    Canvas {
                                        id: npuGaugeCanvas
                                        width: 110; height: 110
                                        onPaint: drawGauge(getContext("2d"), width, height, sysMonApp.npuVal, "#34D399")
                                    }
                                    Text {
                                        text: Math.round(sysMonApp.npuVal) + "%"
                                        color: "#34D399"
                                        font.pixelSize: 22; font.bold: true; font.family: Theme.fontFamily
                                    }
                                    Text {
                                        text: "NPU"
                                        color: Theme.textDim
                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                    }
                                }
                            }
                        }

                        // ── Key stats row ────────────────────
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            Rectangle {
                                Layout.fillWidth: true; Layout.preferredHeight: 64
                                radius: Theme.radiusTiny; color: Theme.surface
                                Column {
                                    anchors.centerIn: parent; spacing: 4
                                    Text { text: "4d 7h 23m"; color: Theme.text; font.pixelSize: 15; font.bold: true; font.family: Theme.fontFamily }
                                    Text { text: "Uptime"; color: Theme.textMuted; font.pixelSize: 11; font.family: Theme.fontFamily }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; Layout.preferredHeight: 64
                                radius: Theme.radiusTiny; color: Theme.surface
                                Column {
                                    anchors.centerIn: parent; spacing: 4
                                    Text { text: "218"; color: Theme.text; font.pixelSize: 15; font.bold: true; font.family: Theme.fontFamily }
                                    Text { text: "Processes"; color: Theme.textMuted; font.pixelSize: 11; font.family: Theme.fontFamily }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; Layout.preferredHeight: 64
                                radius: Theme.radiusTiny; color: Theme.surface
                                Column {
                                    anchors.centerIn: parent; spacing: 4
                                    Text { text: "1,347"; color: Theme.text; font.pixelSize: 15; font.bold: true; font.family: Theme.fontFamily }
                                    Text { text: "Threads"; color: Theme.textMuted; font.pixelSize: 11; font.family: Theme.fontFamily }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; Layout.preferredHeight: 64
                                radius: Theme.radiusTiny; color: Theme.surface
                                Column {
                                    anchors.centerIn: parent; spacing: 4
                                    Text { text: "1.2 Gbps"; color: Theme.text; font.pixelSize: 15; font.bold: true; font.family: Theme.fontFamily }
                                    Text { text: "Network"; color: Theme.textMuted; font.pixelSize: 11; font.family: Theme.fontFamily }
                                }
                            }
                        }

                        // ── System info card ─────────────────
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 80
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 16
                                spacing: 32

                                Column {
                                    spacing: 2
                                    Text { text: "Hostname"; color: Theme.textMuted; font.pixelSize: 10; font.family: Theme.fontFamily }
                                    Text { text: "neuraos-workstation"; color: Theme.text; font.pixelSize: 13; font.bold: true; font.family: Theme.fontFamily }
                                }
                                Column {
                                    spacing: 2
                                    Text { text: "Kernel"; color: Theme.textMuted; font.pixelSize: 10; font.family: Theme.fontFamily }
                                    Text { text: "NeuralOS 4.0.12-npu"; color: Theme.text; font.pixelSize: 13; font.bold: true; font.family: Theme.fontFamily }
                                }
                                Column {
                                    spacing: 2
                                    Text { text: "Architecture"; color: Theme.textMuted; font.pixelSize: 10; font.family: Theme.fontFamily }
                                    Text { text: "aarch64 + NPU v3"; color: Theme.text; font.pixelSize: 13; font.bold: true; font.family: Theme.fontFamily }
                                }
                                Item { Layout.fillWidth: true }
                            }
                        }

                        Item { Layout.preferredHeight: 8 }
                    }
                }

                // ========== PROCESSES TAB ==========
                Item {
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12
                        spacing: 10

                        // Table card
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: Theme.radiusSmall
                            color: Theme.surface
                            clip: true

                            Column {
                                anchors.fill: parent; anchors.margins: 8
                                spacing: 0

                                // Table header
                                Row {
                                    width: parent.width; height: 32
                                    Text { width: 70; height: 32; verticalAlignment: Text.AlignVCenter; text: "PID"; color: Theme.textDim; font.pixelSize: 11; font.bold: true; font.family: Theme.fontFamily }
                                    Text { width: parent.width - 320; height: 32; verticalAlignment: Text.AlignVCenter; text: "Name"; color: Theme.textDim; font.pixelSize: 11; font.bold: true; font.family: Theme.fontFamily }
                                    Text { width: 70; height: 32; verticalAlignment: Text.AlignVCenter; text: "CPU%"; color: Theme.textDim; font.pixelSize: 11; font.bold: true; font.family: Theme.fontFamily }
                                    Text { width: 90; height: 32; verticalAlignment: Text.AlignVCenter; text: "Mem"; color: Theme.textDim; font.pixelSize: 11; font.bold: true; font.family: Theme.fontFamily }
                                    Text { width: 90; height: 32; verticalAlignment: Text.AlignVCenter; text: "Status"; color: Theme.textDim; font.pixelSize: 11; font.bold: true; font.family: Theme.fontFamily }
                                }

                                Rectangle { width: parent.width; height: 1; color: Theme.surfaceLight }

                                // Process rows via ListView
                                ListView {
                                    width: parent.width
                                    height: parent.height - 33
                                    clip: true
                                    model: sysMonApp.processList

                                    delegate: Rectangle {
                                        width: ListView.view.width
                                        height: 30
                                        color: index % 2 === 0 ? "transparent" : Theme.surfaceAlt
                                        radius: Theme.radiusTiny

                                        Row {
                                            anchors.fill: parent
                                            spacing: 0

                                            Text {
                                                width: 70; height: 30
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.pid
                                                color: Theme.textDim
                                                font.pixelSize: 11; font.family: Theme.fontFamily
                                            }
                                            Text {
                                                width: parent.width - 320; height: 30
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.name
                                                color: Theme.text
                                                font.pixelSize: 11; font.family: Theme.fontFamily
                                                elide: Text.ElideRight
                                            }
                                            Text {
                                                width: 70; height: 30
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.cpu + "%"
                                                color: parseFloat(modelData.cpu) > 3 ? Theme.warning : Theme.textDim
                                                font.pixelSize: 11; font.family: Theme.fontFamily
                                            }
                                            Text {
                                                width: 90; height: 30
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.mem
                                                color: Theme.textDim
                                                font.pixelSize: 11; font.family: Theme.fontFamily
                                            }
                                            Text {
                                                width: 90; height: 30
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.status
                                                color: modelData.status === "Running" ? Theme.success
                                                     : modelData.status === "Zombie"  ? Theme.error
                                                     : Theme.textMuted
                                                font.pixelSize: 11; font.family: Theme.fontFamily
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ========== PERFORMANCE TAB ==========
                Flickable {
                    contentHeight: perfCol.height + 24
                    clip: true

                    ColumnLayout {
                        id: perfCol
                        width: parent.width - 24
                        x: 12; y: 12
                        spacing: 14

                        // CPU graph card
                        Rectangle {
                            id: cpuCard
                            Layout.fillWidth: true
                            Layout.preferredHeight: 250
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            Text {
                                x: 14; y: 14
                                text: "CPU Usage  -  " + Math.round(sysMonApp.cpuVal) + "%"
                                color: "#5B9AFF"
                                font.pixelSize: 13; font.bold: true; font.family: Theme.fontFamily
                            }
                            Text {
                                x: cpuCard.width - implicitWidth - 14; y: 14
                                text: "60s window"
                                color: Theme.textMuted
                                font.pixelSize: 10; font.family: Theme.fontFamily
                            }
                            Canvas {
                                id: cpuGraphCanvas
                                x: 10; y: 36
                                width: cpuCard.width - 20; height: cpuCard.height - 46
                                onPaint: drawGraph(getContext("2d"), width, height, sysMonApp.cpuHistory, "#5B9AFF")
                            }
                        }

                        // Memory graph card
                        Rectangle {
                            id: memCard
                            Layout.fillWidth: true
                            Layout.preferredHeight: 250
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            Text {
                                x: 14; y: 14
                                text: "Memory Usage  -  " + Math.round(sysMonApp.memVal) + "%"
                                color: "#A78BFA"
                                font.pixelSize: 13; font.bold: true; font.family: Theme.fontFamily
                            }
                            Text {
                                x: memCard.width - implicitWidth - 14; y: 14
                                text: "60s window"
                                color: Theme.textMuted
                                font.pixelSize: 10; font.family: Theme.fontFamily
                            }
                            Canvas {
                                id: memGraphCanvas
                                x: 10; y: 36
                                width: memCard.width - 20; height: memCard.height - 46
                                onPaint: drawGraph(getContext("2d"), width, height, sysMonApp.memHistory, "#A78BFA")
                            }
                        }

                        Item { Layout.preferredHeight: 8 }
                    }
                }
            }
        }
    }
}
