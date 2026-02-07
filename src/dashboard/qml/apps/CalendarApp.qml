import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: calApp
    anchors.fill: parent

    property int currentYear: new Date().getFullYear()
    property int currentMonth: new Date().getMonth()
    property int selectedDay: new Date().getDate()
    property int todayYear: new Date().getFullYear()
    property int todayMonth: new Date().getMonth()
    property int todayDay: new Date().getDate()

    property var monthNames: ["January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December"]
    property var dayNames: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    property var events: [
        { date: "2026-02-07", title: "v3.4 Planning", time: "10:00", color: "#4C8FFF" },
        { date: "2026-02-07", title: "Team Standup", time: "09:00", color: "#10B981" },
        { date: "2026-02-10", title: "Sprint Review", time: "14:00", color: "#F59E0B" },
        { date: "2026-02-14", title: "Release Day", time: "00:00", color: "#E84855" },
        { date: "2026-02-20", title: "Security Audit", time: "11:00", color: "#7C3AED" },
        { date: "2026-02-25", title: "Team Retrospective", time: "15:00", color: "#EC4899" },
        { date: "2026-03-01", title: "Q1 Review", time: "10:00", color: "#4C8FFF" },
        { date: "2026-01-15", title: "Board Meeting", time: "09:00", color: "#F59E0B" }
    ]

    function getDaysInMonth(year, month) {
        return new Date(year, month + 1, 0).getDate()
    }

    function getFirstDayOfWeek(year, month) {
        var d = new Date(year, month, 1).getDay()
        return d === 0 ? 6 : d - 1
    }

    function generateCalendarDays() {
        var days = []
        var firstDay = getFirstDayOfWeek(currentYear, currentMonth)
        var daysInMonth = getDaysInMonth(currentYear, currentMonth)
        var prevMonth = currentMonth === 0 ? 11 : currentMonth - 1
        var prevYear = currentMonth === 0 ? currentYear - 1 : currentYear
        var prevDays = getDaysInMonth(prevYear, prevMonth)

        for (var p = firstDay - 1; p >= 0; p--)
            days.push({ day: prevDays - p, currentMonth: false })
        for (var d = 1; d <= daysInMonth; d++)
            days.push({ day: d, currentMonth: true })
        var remaining = 42 - days.length
        for (var n = 1; n <= remaining; n++)
            days.push({ day: n, currentMonth: false })

        return days
    }

    function getEventsForDate(day) {
        var m = (currentMonth + 1).toString()
        if (m.length === 1) m = "0" + m
        var dd = day.toString()
        if (dd.length === 1) dd = "0" + dd
        var dateStr = currentYear + "-" + m + "-" + dd
        return events.filter(function(e) { return e.date === dateStr })
    }

    function hasEvents(day) {
        return getEventsForDate(day).length > 0
    }

    function prevMonth() {
        if (currentMonth === 0) { currentMonth = 11; currentYear-- }
        else currentMonth--
        selectedDay = 1
    }

    function nextMonth() {
        if (currentMonth === 11) { currentMonth = 0; currentYear++ }
        else currentMonth++
        selectedDay = 1
    }

    function isToday(day) {
        return day === todayDay && currentMonth === todayMonth && currentYear === todayYear
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Left: Calendar Grid ─── */
            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                spacing: 0

                /* Month/Year Navigation */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 56
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 16; anchors.rightMargin: 16; spacing: 12

                        Rectangle {
                            width: 32; height: 32; radius: Theme.radiusSmall
                            color: prevMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Components.CanvasIcon { anchors.centerIn: parent; iconName: "arrow-left"; iconSize: 14; iconColor: Theme.textDim }
                            MouseArea { id: prevMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: calApp.prevMonth() }
                        }

                        Text {
                            Layout.fillWidth: true
                            text: monthNames[currentMonth] + " " + currentYear
                            font.pixelSize: 18; font.bold: true; color: Theme.text
                            horizontalAlignment: Text.AlignHCenter
                        }

                        Rectangle {
                            width: 32; height: 32; radius: Theme.radiusSmall
                            color: nextMa.containsMouse ? Theme.surfaceAlt : "transparent"
                            Components.CanvasIcon { anchors.centerIn: parent; iconName: "arrow-right"; iconSize: 14; iconColor: Theme.textDim }
                            MouseArea { id: nextMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: calApp.nextMonth() }
                        }

                        Rectangle {
                            width: todayLbl.implicitWidth + 16; height: 28; radius: 14
                            color: todayBtnMa.containsMouse ? Theme.primary : Theme.surfaceAlt
                            Text { id: todayLbl; anchors.centerIn: parent; text: "Today"; font.pixelSize: 11; color: todayBtnMa.containsMouse ? "#FFFFFF" : Theme.textDim }
                            MouseArea {
                                id: todayBtnMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: { currentYear = todayYear; currentMonth = todayMonth; selectedDay = todayDay }
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                /* Day-of-week Headers */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 32
                    color: Theme.surfaceAlt

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 8; anchors.rightMargin: 8; spacing: 0

                        Repeater {
                            model: dayNames
                            Text {
                                Layout.fillWidth: true; text: modelData
                                font.pixelSize: 11; font.weight: Font.DemiBold
                                color: index >= 5 ? Theme.primary : Theme.textDim
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }
                    }
                }

                /* Calendar Grid */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: Theme.background

                    GridLayout {
                        anchors.fill: parent; anchors.margins: 8
                        columns: 7; rowSpacing: 4; columnSpacing: 4

                        Repeater {
                            model: generateCalendarDays()

                            Rectangle {
                                Layout.fillWidth: true; Layout.fillHeight: true
                                radius: Theme.radiusSmall
                                color: {
                                    if (modelData.currentMonth && modelData.day === selectedDay)
                                        return Theme.primary
                                    if (modelData.currentMonth && isToday(modelData.day))
                                        return Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.15)
                                    if (dayMa.containsMouse && modelData.currentMonth)
                                        return Theme.hoverBg
                                    return "transparent"
                                }
                                border.width: modelData.currentMonth && isToday(modelData.day) && modelData.day !== selectedDay ? 1 : 0
                                border.color: Theme.primary

                                ColumnLayout {
                                    anchors.fill: parent; anchors.margins: 4; spacing: 2

                                    Text {
                                        Layout.alignment: Qt.AlignHCenter
                                        text: modelData.day
                                        font.pixelSize: 13
                                        font.bold: modelData.currentMonth && (isToday(modelData.day) || modelData.day === selectedDay)
                                        color: {
                                            if (modelData.currentMonth && modelData.day === selectedDay) return "#FFFFFF"
                                            if (!modelData.currentMonth) return Theme.textMuted
                                            if (isToday(modelData.day)) return Theme.primary
                                            return Theme.text
                                        }
                                    }

                                    /* Event dots */
                                    Row {
                                        Layout.alignment: Qt.AlignHCenter
                                        spacing: 3
                                        visible: modelData.currentMonth && hasEvents(modelData.day)

                                        Repeater {
                                            model: {
                                                if (!modelData.currentMonth) return []
                                                var evts = getEventsForDate(modelData.day)
                                                return evts.length > 3 ? evts.slice(0, 3) : evts
                                            }

                                            Rectangle {
                                                width: 5; height: 5; radius: 3
                                                color: modelData.day === selectedDay ? "#FFFFFF" : modelData.color
                                            }
                                        }
                                    }

                                    Item { Layout.fillHeight: true }
                                }

                                MouseArea {
                                    id: dayMa; anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: modelData.currentMonth ? Qt.PointingHandCursor : Qt.ArrowCursor
                                    onClicked: if (modelData.currentMonth) selectedDay = modelData.day
                                }
                            }
                        }
                    }
                }
            }

            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

            /* ─── Right: Events Panel ─── */
            Rectangle {
                Layout.preferredWidth: 240; Layout.fillHeight: true
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent; spacing: 0

                    /* Selected Date Header */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 90
                        color: "transparent"

                        ColumnLayout {
                            anchors.centerIn: parent; spacing: 4

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: selectedDay
                                font.pixelSize: 36; font.bold: true
                                color: Theme.primary
                            }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: monthNames[currentMonth] + " " + currentYear
                                font.pixelSize: 12; color: Theme.textDim
                            }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: {
                                    var d = new Date(currentYear, currentMonth, selectedDay)
                                    var days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                                    return days[d.getDay()]
                                }
                                font.pixelSize: 11; color: Theme.textMuted
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; Layout.leftMargin: 12; Layout.rightMargin: 12; height: 1; color: Theme.surfaceLight }

                    /* Events Header */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 36
                        color: "transparent"

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14

                            Text {
                                Layout.fillWidth: true; text: "Events"
                                font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text
                            }

                            Text {
                                text: getEventsForDate(selectedDay).length.toString()
                                font.pixelSize: 11; color: Theme.textMuted
                            }
                        }
                    }

                    /* Events List */
                    ListView {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        clip: true; model: getEventsForDate(selectedDay); spacing: 6

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded; width: 4
                            contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                        }

                        delegate: Rectangle {
                            width: ListView.view ? ListView.view.width - 24 : 0; height: 56
                            x: 12; radius: Theme.radiusSmall
                            color: Theme.surfaceAlt

                            RowLayout {
                                anchors.fill: parent; anchors.margins: 10; spacing: 10

                                Rectangle {
                                    width: 4; height: parent.height - 8; radius: 2
                                    anchors.verticalCenter: parent.verticalCenter
                                    color: modelData.color
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true; spacing: 3

                                    Text {
                                        text: modelData.title; font.pixelSize: 12
                                        font.weight: Font.DemiBold; color: Theme.text
                                    }

                                    RowLayout {
                                        spacing: 4
                                        Components.CanvasIcon { iconName: "clock"; iconSize: 10; iconColor: Theme.textMuted }
                                        Text { text: modelData.time; font.pixelSize: 10; color: Theme.textMuted }
                                    }
                                }
                            }
                        }

                        /* Empty state */
                        Text {
                            visible: getEventsForDate(selectedDay).length === 0
                            anchors.centerIn: parent
                            text: "No events"
                            font.pixelSize: 12; color: Theme.textMuted
                        }
                    }

                    /* Mini info */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 28
                        color: Theme.surfaceAlt

                        Text {
                            anchors.centerIn: parent
                            text: events.length + " total events"
                            font.pixelSize: 10; color: Theme.textDim
                        }
                    }
                }
            }
        }
    }
}
