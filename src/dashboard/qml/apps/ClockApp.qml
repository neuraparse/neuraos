import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: clockApp
    anchors.fill: parent

    property int currentTab: 0
    property var tabNames: ["Clock", "Alarm", "Stopwatch", "Timer"]

    /* Stopwatch state */
    property bool swRunning: false
    property int swElapsed: 0
    property var swLaps: []

    /* Timer state */
    property bool tmRunning: false
    property int tmTotal: 60
    property int tmRemaining: 60

    /* World clock offsets (hours from UTC) */
    property var worldClocks: [
        { city: "New York",  offset: -5, flag: "US" },
        { city: "London",    offset:  0, flag: "GB" },
        { city: "Tokyo",     offset:  9, flag: "JP" },
        { city: "Sydney",    offset: 11, flag: "AU" }
    ]

    /* Clock tick timer */
    Timer {
        id: clockTimer
        interval: 1000; running: true; repeat: true
        onTriggered: analogClock.requestPaint()
    }

    /* Stopwatch timer */
    Timer {
        id: swTimer
        interval: 10; running: swRunning; repeat: true
        onTriggered: swElapsed += 10
    }

    /* Countdown timer */
    Timer {
        id: tmTimer
        interval: 1000; running: tmRunning; repeat: true
        onTriggered: {
            if (tmRemaining > 0) tmRemaining--
            else { tmRunning = false }
            timerRing.requestPaint()
        }
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
                Layout.preferredHeight: 42
                color: Theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12; anchors.rightMargin: 12
                    spacing: 4

                    Repeater {
                        model: tabNames

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.topMargin: 6; Layout.bottomMargin: 6
                            radius: Theme.radiusTiny
                            color: currentTab === index ? Theme.primary
                                 : tabMa.containsMouse ? Theme.surfaceAlt : "transparent"

                            Text {
                                anchors.centerIn: parent
                                text: modelData
                                font.pixelSize: 12
                                font.weight: currentTab === index ? Font.DemiBold : Font.Normal
                                font.family: Theme.fontFamily
                                color: currentTab === index ? "#FFFFFF" : Theme.textDim
                            }

                            MouseArea {
                                id: tabMa
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: currentTab = index
                            }

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

            /* Tab Content */
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: currentTab

                /* ─── Clock Tab ─── */
                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Item { Layout.fillHeight: true; Layout.maximumHeight: 20 }

                        /* Analog Clock */
                        Canvas {
                            id: analogClock
                            Layout.alignment: Qt.AlignHCenter
                            Layout.preferredWidth: 220
                            Layout.preferredHeight: 220

                            onPaint: {
                                var ctx = getContext("2d")
                                var w = width, h = height, cx = w / 2, cy = h / 2, r = Math.min(cx, cy) - 8
                                var now = new Date()
                                var hrs = now.getHours(), min = now.getMinutes(), sec = now.getSeconds()

                                ctx.clearRect(0, 0, w, h)

                                /* Face */
                                ctx.beginPath()
                                ctx.arc(cx, cy, r, 0, Math.PI * 2)
                                ctx.fillStyle = String(Theme.surface)
                                ctx.fill()
                                ctx.lineWidth = 2
                                ctx.strokeStyle = String(Theme.surfaceLight)
                                ctx.stroke()

                                /* Tick marks */
                                for (var i = 0; i < 60; i++) {
                                    var angle = (i * 6 - 90) * Math.PI / 180
                                    var isMajor = i % 5 === 0
                                    var outerR = r - 4
                                    var innerR = isMajor ? r - 16 : r - 10
                                    ctx.beginPath()
                                    ctx.moveTo(cx + Math.cos(angle) * innerR, cy + Math.sin(angle) * innerR)
                                    ctx.lineTo(cx + Math.cos(angle) * outerR, cy + Math.sin(angle) * outerR)
                                    ctx.lineWidth = isMajor ? 2 : 1
                                    ctx.strokeStyle = isMajor ? String(Theme.text) : String(Theme.textMuted)
                                    ctx.stroke()
                                }

                                /* Hour numbers */
                                ctx.font = "bold 12px " + Theme.fontFamily
                                ctx.fillStyle = String(Theme.text)
                                ctx.textAlign = "center"
                                ctx.textBaseline = "middle"
                                for (var n = 1; n <= 12; n++) {
                                    var na = (n * 30 - 90) * Math.PI / 180
                                    var nr = r - 28
                                    ctx.fillText(n.toString(), cx + Math.cos(na) * nr, cy + Math.sin(na) * nr)
                                }

                                /* Hour hand */
                                var hAngle = ((hrs % 12) * 30 + min * 0.5 - 90) * Math.PI / 180
                                ctx.beginPath()
                                ctx.moveTo(cx, cy)
                                ctx.lineTo(cx + Math.cos(hAngle) * (r * 0.5), cy + Math.sin(hAngle) * (r * 0.5))
                                ctx.lineWidth = 4; ctx.lineCap = "round"
                                ctx.strokeStyle = String(Theme.text)
                                ctx.stroke()

                                /* Minute hand */
                                var mAngle = (min * 6 + sec * 0.1 - 90) * Math.PI / 180
                                ctx.beginPath()
                                ctx.moveTo(cx, cy)
                                ctx.lineTo(cx + Math.cos(mAngle) * (r * 0.7), cy + Math.sin(mAngle) * (r * 0.7))
                                ctx.lineWidth = 3; ctx.lineCap = "round"
                                ctx.strokeStyle = String(Theme.primary)
                                ctx.stroke()

                                /* Second hand */
                                var sAngle = (sec * 6 - 90) * Math.PI / 180
                                ctx.beginPath()
                                ctx.moveTo(cx, cy)
                                ctx.lineTo(cx + Math.cos(sAngle) * (r * 0.78), cy + Math.sin(sAngle) * (r * 0.78))
                                ctx.lineWidth = 1.5; ctx.lineCap = "round"
                                ctx.strokeStyle = String(Theme.error)
                                ctx.stroke()

                                /* Center dot */
                                ctx.beginPath()
                                ctx.arc(cx, cy, 4, 0, Math.PI * 2)
                                ctx.fillStyle = String(Theme.error)
                                ctx.fill()
                            }

                            Component.onCompleted: requestPaint()
                        }

                        /* Digital time */
                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: Qt.formatTime(new Date(), "hh:mm:ss")
                            font.pixelSize: 32; font.weight: Font.Light
                            font.family: Theme.fontFamily
                            color: Theme.text

                            Timer { interval: 1000; running: true; repeat: true; onTriggered: parent.text = Qt.formatTime(new Date(), "hh:mm:ss") }
                        }

                        /* Date */
                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: Qt.formatDate(new Date(), "dddd, MMMM d, yyyy")
                            font.pixelSize: 13; font.family: Theme.fontFamily
                            color: Theme.textDim
                        }

                        Item { Layout.preferredHeight: 8 }

                        /* World Clocks */
                        Rectangle {
                            Layout.fillWidth: true; Layout.leftMargin: 20; Layout.rightMargin: 20
                            Layout.preferredHeight: worldCol.height + 20
                            radius: Theme.radiusSmall
                            color: Theme.surface

                            Column {
                                id: worldCol
                                anchors.left: parent.left; anchors.right: parent.right
                                anchors.top: parent.top; anchors.margins: 10
                                spacing: 6

                                Repeater {
                                    model: worldClocks

                                    RowLayout {
                                        width: parent.width; spacing: 8

                                        Text {
                                            text: modelData.flag
                                            font.pixelSize: 14
                                        }

                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.city
                                            font.pixelSize: 12; font.family: Theme.fontFamily
                                            color: Theme.text
                                        }

                                        Text {
                                            text: {
                                                var d = new Date()
                                                var utc = d.getTime() + d.getTimezoneOffset() * 60000
                                                var city = new Date(utc + modelData.offset * 3600000)
                                                return Qt.formatTime(city, "hh:mm")
                                            }
                                            font.pixelSize: 13; font.weight: Font.DemiBold
                                            font.family: Theme.fontFamily
                                            color: Theme.primary
                                        }
                                    }
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }

                /* ─── Alarm Tab ─── */
                Item {
                    id: alarmTab
                    property var alarms: [
                        { time: "06:30", label: "Wake Up", enabled: true, days: "Mon-Fri" },
                        { time: "07:45", label: "Morning Run", enabled: false, days: "Mon, Wed, Fri" },
                        { time: "12:00", label: "Lunch Break", enabled: true, days: "Daily" },
                        { time: "22:00", label: "Wind Down", enabled: true, days: "Daily" }
                    ]

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 8

                        Text {
                            text: "Alarms"
                            font.pixelSize: 18; font.weight: Font.DemiBold
                            font.family: Theme.fontFamily; color: Theme.text
                        }

                        Flickable {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            contentHeight: alarmCol.height; clip: true

                            Column {
                                id: alarmCol
                                width: parent.width; spacing: 8

                                Repeater {
                                    model: alarmTab.alarms

                                    Rectangle {
                                        width: parent.width; height: 72
                                        radius: Theme.radiusSmall; color: Theme.surface

                                        RowLayout {
                                            anchors.fill: parent; anchors.margins: 14; spacing: 12

                                            ColumnLayout {
                                                Layout.fillWidth: true; spacing: 2

                                                Text {
                                                    text: modelData.time
                                                    font.pixelSize: 28; font.weight: Font.Light
                                                    font.family: Theme.fontFamily
                                                    color: modelData.enabled ? Theme.text : Theme.textMuted
                                                }

                                                RowLayout {
                                                    spacing: 6
                                                    Text {
                                                        text: modelData.label
                                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                                        color: Theme.textDim
                                                    }
                                                    Text {
                                                        text: "·  " + modelData.days
                                                        font.pixelSize: 11; font.family: Theme.fontFamily
                                                        color: Theme.textMuted
                                                    }
                                                }
                                            }

                                            Components.ToggleSwitch {
                                                checked: modelData.enabled
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        /* Add Alarm Button */
                        Rectangle {
                            Layout.fillWidth: true; Layout.preferredHeight: 44
                            radius: Theme.radiusSmall
                            color: addMa.containsMouse ? Qt.lighter(Theme.primary, 1.1) : Theme.primary

                            RowLayout {
                                anchors.centerIn: parent; spacing: 6
                                Components.CanvasIcon { iconName: "plus"; iconSize: 16; iconColor: "#FFFFFF" }
                                Text {
                                    text: "Add Alarm"
                                    font.pixelSize: 13; font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily; color: "#FFFFFF"
                                }
                            }

                            MouseArea {
                                id: addMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            }

                            Behavior on color { ColorAnimation { duration: Theme.animFast } }
                        }
                    }
                }

                /* ─── Stopwatch Tab ─── */
                Item {
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 16

                        Item { Layout.fillHeight: true; Layout.maximumHeight: 30 }

                        /* Large display */
                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: {
                                var mins = Math.floor(swElapsed / 60000)
                                var secs = Math.floor((swElapsed % 60000) / 1000)
                                var cs   = Math.floor((swElapsed % 1000) / 10)
                                return (mins < 10 ? "0" : "") + mins + ":" +
                                       (secs < 10 ? "0" : "") + secs + "." +
                                       (cs < 10 ? "0" : "") + cs
                            }
                            font.pixelSize: 56; font.weight: Font.Light
                            font.family: Theme.fontFamily; color: Theme.text
                        }

                        /* Controls */
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter; spacing: 12

                            Rectangle {
                                width: 90; height: 38; radius: Theme.radiusSmall
                                color: swRunning ? Theme.surfaceAlt : Theme.success
                                Text {
                                    anchors.centerIn: parent
                                    text: swRunning ? "Lap" : "Start"
                                    font.pixelSize: 13; font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily
                                    color: swRunning ? Theme.text : "#FFFFFF"
                                }
                                MouseArea {
                                    anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (swRunning) { swLaps = [swElapsed].concat(swLaps); swLapsChanged() }
                                        else swRunning = true
                                    }
                                }
                            }

                            Rectangle {
                                width: 90; height: 38; radius: Theme.radiusSmall
                                color: swRunning ? Theme.error : Theme.surfaceAlt
                                Text {
                                    anchors.centerIn: parent
                                    text: swRunning ? "Stop" : "Reset"
                                    font.pixelSize: 13; font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily
                                    color: swRunning ? "#FFFFFF" : Theme.textDim
                                }
                                MouseArea {
                                    anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (swRunning) swRunning = false
                                        else { swElapsed = 0; swLaps = []; swLapsChanged() }
                                    }
                                }
                            }
                        }

                        /* Lap List */
                        Rectangle {
                            Layout.fillWidth: true; Layout.fillHeight: true
                            Layout.leftMargin: 20; Layout.rightMargin: 20
                            radius: Theme.radiusSmall; color: Theme.surface; clip: true

                            Flickable {
                                anchors.fill: parent; anchors.margins: 8
                                contentHeight: lapCol.height; clip: true

                                Column {
                                    id: lapCol; width: parent.width; spacing: 4

                                    Repeater {
                                        model: swLaps.length

                                        RowLayout {
                                            width: parent.width; height: 30

                                            Text {
                                                Layout.fillWidth: true
                                                text: "Lap " + (swLaps.length - index)
                                                font.pixelSize: 12; font.family: Theme.fontFamily
                                                color: Theme.textDim
                                            }

                                            Text {
                                                text: {
                                                    var e = swLaps[index]
                                                    var m = Math.floor(e / 60000)
                                                    var s = Math.floor((e % 60000) / 1000)
                                                    var c = Math.floor((e % 1000) / 10)
                                                    return (m < 10 ? "0" : "") + m + ":" +
                                                           (s < 10 ? "0" : "") + s + "." +
                                                           (c < 10 ? "0" : "") + c
                                                }
                                                font.pixelSize: 13; font.weight: Font.DemiBold
                                                font.family: Theme.fontFamily
                                                color: Theme.text
                                            }
                                        }
                                    }
                                }
                            }

                            Text {
                                anchors.centerIn: parent
                                visible: swLaps.length === 0
                                text: "No laps recorded"
                                font.pixelSize: 12; font.family: Theme.fontFamily
                                color: Theme.textMuted
                            }
                        }
                    }
                }

                /* ─── Timer Tab ─── */
                Item {
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 16; spacing: 14

                        Item { Layout.fillHeight: true; Layout.maximumHeight: 16 }

                        /* Progress ring */
                        Canvas {
                            id: timerRing
                            Layout.alignment: Qt.AlignHCenter
                            Layout.preferredWidth: 200; Layout.preferredHeight: 200

                            onPaint: {
                                var ctx = getContext("2d")
                                var w = width, h = height, cx = w / 2, cy = h / 2, r = 85
                                ctx.clearRect(0, 0, w, h)

                                /* Background ring */
                                ctx.beginPath()
                                ctx.arc(cx, cy, r, 0, Math.PI * 2)
                                ctx.lineWidth = 8
                                ctx.strokeStyle = String(Theme.surfaceLight)
                                ctx.stroke()

                                /* Progress arc */
                                var frac = tmTotal > 0 ? tmRemaining / tmTotal : 0
                                if (frac > 0) {
                                    ctx.beginPath()
                                    ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2)
                                    ctx.lineWidth = 8; ctx.lineCap = "round"
                                    ctx.strokeStyle = String(tmRemaining <= 10 ? Theme.error : Theme.primary)
                                    ctx.stroke()
                                }

                                /* Center time text */
                                var mins = Math.floor(tmRemaining / 60)
                                var secs = tmRemaining % 60
                                var txt = (mins < 10 ? "0" : "") + mins + ":" + (secs < 10 ? "0" : "") + secs
                                ctx.font = "300 36px " + Theme.fontFamily
                                ctx.fillStyle = String(Theme.text)
                                ctx.textAlign = "center"; ctx.textBaseline = "middle"
                                ctx.fillText(txt, cx, cy)
                            }

                            Component.onCompleted: requestPaint()
                        }

                        /* Presets */
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter; spacing: 8

                            Repeater {
                                model: [
                                    { label: "1 min", secs: 60 },
                                    { label: "5 min", secs: 300 },
                                    { label: "10 min", secs: 600 },
                                    { label: "30 min", secs: 1800 }
                                ]

                                Rectangle {
                                    width: 64; height: 30; radius: Theme.radiusTiny
                                    color: tmTotal === modelData.secs && !tmRunning
                                         ? Theme.primary
                                         : presetMa.containsMouse ? Theme.surfaceAlt : Theme.surface

                                    Text {
                                        anchors.centerIn: parent; text: modelData.label
                                        font.pixelSize: 11; font.weight: Font.Medium
                                        font.family: Theme.fontFamily
                                        color: tmTotal === modelData.secs && !tmRunning ? "#FFFFFF" : Theme.textDim
                                    }

                                    MouseArea {
                                        id: presetMa; anchors.fill: parent
                                        hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            tmRunning = false
                                            tmTotal = modelData.secs
                                            tmRemaining = modelData.secs
                                            timerRing.requestPaint()
                                        }
                                    }
                                }
                            }
                        }

                        /* Timer controls */
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter; spacing: 12

                            Rectangle {
                                width: 100; height: 40; radius: Theme.radiusSmall
                                color: tmStartMa.containsMouse ? Qt.lighter(Theme.primary, 1.1) : Theme.primary
                                Text {
                                    anchors.centerIn: parent
                                    text: tmRunning ? "Pause" : "Start"
                                    font.pixelSize: 13; font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily; color: "#FFFFFF"
                                }
                                MouseArea {
                                    id: tmStartMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (tmRemaining > 0) tmRunning = !tmRunning
                                    }
                                }
                                Behavior on color { ColorAnimation { duration: Theme.animFast } }
                            }

                            Rectangle {
                                width: 100; height: 40; radius: Theme.radiusSmall
                                color: tmResetMa.containsMouse ? Theme.surfaceAlt : Theme.surface
                                Text {
                                    anchors.centerIn: parent; text: "Reset"
                                    font.pixelSize: 13; font.weight: Font.DemiBold
                                    font.family: Theme.fontFamily; color: Theme.textDim
                                }
                                MouseArea {
                                    id: tmResetMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        tmRunning = false
                                        tmRemaining = tmTotal
                                        timerRing.requestPaint()
                                    }
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }
                    }
                }
            }
        }
    }
}
