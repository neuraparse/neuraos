import QtQuick 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: weatherApp
    anchors.fill: parent

    property string condition: "Partly Cloudy"
    property int currentTemp: 72
    property string location: "San Francisco, CA"
    property int feelsLike: 74
    property int humidity: 65
    property int windSpeed: 12
    property int uvIndex: 6
    property int pressure: 1013
    property int visibility: 10

    property var hourlyForecast: [
        { time: "Now",  icon: "sun",   temp: 72 },
        { time: "1 PM", icon: "sun",   temp: 74 },
        { time: "2 PM", icon: "cloud", temp: 73 },
        { time: "3 PM", icon: "cloud", temp: 71 },
        { time: "4 PM", icon: "rain",  temp: 68 },
        { time: "5 PM", icon: "rain",  temp: 66 },
        { time: "6 PM", icon: "cloud", temp: 65 },
        { time: "7 PM", icon: "cloud", temp: 63 }
    ]

    property var weeklyForecast: [
        { day: "Today",     icon: "sun",   high: 74, low: 58 },
        { day: "Tuesday",   icon: "cloud", high: 70, low: 56 },
        { day: "Wednesday", icon: "rain",  high: 64, low: 52 },
        { day: "Thursday",  icon: "rain",  high: 62, low: 50 },
        { day: "Friday",    icon: "cloud", high: 67, low: 54 },
        { day: "Saturday",  icon: "sun",   high: 72, low: 57 },
        { day: "Sunday",    icon: "snow",  high: 48, low: 35 }
    ]

    /* Draw a weather icon on a Canvas context */
    function drawWeatherIcon(ctx, iconType, cx, cy, size) {
        ctx.save()
        if (iconType === "sun") {
            /* Sun circle */
            ctx.beginPath()
            ctx.arc(cx, cy, size * 0.32, 0, Math.PI * 2)
            ctx.fillStyle = "#FBBF24"
            ctx.fill()
            /* Rays */
            ctx.strokeStyle = "#FBBF24"
            ctx.lineWidth = size * 0.06
            ctx.lineCap = "round"
            for (var r = 0; r < 8; r++) {
                var angle = (r / 8) * Math.PI * 2
                var inner = size * 0.42
                var outer = size * 0.55
                ctx.beginPath()
                ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner)
                ctx.lineTo(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer)
                ctx.stroke()
            }
        } else if (iconType === "cloud") {
            ctx.fillStyle = "#94A3B8"
            ctx.beginPath()
            ctx.arc(cx - size * 0.12, cy + size * 0.05, size * 0.22, Math.PI, 2 * Math.PI)
            ctx.arc(cx + size * 0.08, cy - size * 0.08, size * 0.28, Math.PI * 1.2, 2 * Math.PI * 0.9)
            ctx.arc(cx + size * 0.28, cy + size * 0.05, size * 0.18, Math.PI * 1.3, 2 * Math.PI * 0.8)
            ctx.closePath()
            ctx.fill()
            ctx.fillRect(cx - size * 0.34, cy + size * 0.05, size * 0.68, size * 0.18)
        } else if (iconType === "rain") {
            /* Cloud */
            ctx.fillStyle = "#64748B"
            ctx.beginPath()
            ctx.arc(cx - size * 0.1, cy - size * 0.08, size * 0.2, Math.PI, 2 * Math.PI)
            ctx.arc(cx + size * 0.1, cy - size * 0.15, size * 0.22, Math.PI * 1.2, 2 * Math.PI * 0.9)
            ctx.arc(cx + size * 0.28, cy - size * 0.08, size * 0.15, Math.PI * 1.3, 2 * Math.PI * 0.8)
            ctx.closePath()
            ctx.fill()
            ctx.fillRect(cx - size * 0.3, cy - size * 0.08, size * 0.62, size * 0.14)
            /* Rain lines */
            ctx.strokeStyle = "#38BDF8"
            ctx.lineWidth = size * 0.05
            ctx.lineCap = "round"
            for (var d = 0; d < 3; d++) {
                var dx = cx - size * 0.15 + d * size * 0.18
                ctx.beginPath()
                ctx.moveTo(dx, cy + size * 0.15)
                ctx.lineTo(dx - size * 0.06, cy + size * 0.35)
                ctx.stroke()
            }
        } else if (iconType === "snow") {
            /* Cloud */
            ctx.fillStyle = "#94A3B8"
            ctx.beginPath()
            ctx.arc(cx - size * 0.1, cy - size * 0.08, size * 0.2, Math.PI, 2 * Math.PI)
            ctx.arc(cx + size * 0.1, cy - size * 0.15, size * 0.22, Math.PI * 1.2, 2 * Math.PI * 0.9)
            ctx.arc(cx + size * 0.28, cy - size * 0.08, size * 0.15, Math.PI * 1.3, 2 * Math.PI * 0.8)
            ctx.closePath()
            ctx.fill()
            ctx.fillRect(cx - size * 0.3, cy - size * 0.08, size * 0.62, size * 0.14)
            /* Snow dots */
            ctx.fillStyle = "#E0E7FF"
            for (var s = 0; s < 4; s++) {
                var sx = cx - size * 0.2 + s * size * 0.15
                var sy = cy + size * 0.2 + (s % 2) * size * 0.1
                ctx.beginPath()
                ctx.arc(sx, sy, size * 0.04, 0, Math.PI * 2)
                ctx.fill()
            }
        }
        ctx.restore()
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: condition === "Partly Cloudy" ? "#3B5998" : "#1E3A5F" }
            GradientStop { position: 1.0; color: Theme.background }
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            /* ---- Location Header ---- */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 42
                color: "transparent"

                RowLayout {
                    anchors.centerIn: parent; spacing: 6

                    Components.CanvasIcon {
                        iconName: "map-pin"; iconSize: 14; iconColor: "#FFFFFF"
                    }
                    Text {
                        text: location
                        font.pixelSize: 14; font.weight: Font.DemiBold
                        font.family: Theme.fontFamily; color: "#FFFFFF"
                    }
                }
            }

            /* ---- Current Weather Display ---- */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 180
                color: "transparent"

                RowLayout {
                    anchors.centerIn: parent; spacing: 24

                    /* Large weather icon */
                    Canvas {
                        id: mainIcon
                        width: 110; height: 110

                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            drawWeatherIcon(ctx, "sun", width / 2, height / 2, width)
                        }
                        Component.onCompleted: requestPaint()
                    }

                    ColumnLayout {
                        spacing: 2

                        Text {
                            text: currentTemp + "\u00B0F"
                            font.pixelSize: 64; font.weight: Font.Light
                            font.family: Theme.fontFamily; color: "#FFFFFF"
                        }
                        Text {
                            text: condition
                            font.pixelSize: 16; font.family: Theme.fontFamily
                            color: Qt.rgba(1, 1, 1, 0.75)
                        }
                        Text {
                            text: "Feels like " + feelsLike + "\u00B0F"
                            font.pixelSize: 12; font.family: Theme.fontFamily
                            color: Qt.rgba(1, 1, 1, 0.55)
                        }
                    }
                }
            }

            /* ---- Hourly Forecast ---- */
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 110
                color: Qt.rgba(Theme.surface.r, Theme.surface.g, Theme.surface.b, 0.4)
                radius: Theme.radiusSmall

                Layout.leftMargin: 12; Layout.rightMargin: 12

                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 8; spacing: 4

                    Text {
                        text: "Hourly Forecast"
                        font.pixelSize: 11; font.weight: Font.DemiBold
                        font.family: Theme.fontFamily; color: Theme.textDim
                    }

                    Flickable {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        contentWidth: hourlyRow.width; clip: true
                        flickableDirection: Flickable.HorizontalFlick

                        Row {
                            id: hourlyRow; spacing: 4

                            Repeater {
                                model: hourlyForecast.length

                                Rectangle {
                                    width: 62; height: 70; radius: Theme.radiusSmall
                                    color: index === 0 ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.2)
                                                       : "transparent"

                                    ColumnLayout {
                                        anchors.centerIn: parent; spacing: 3

                                        Text {
                                            Layout.alignment: Qt.AlignHCenter
                                            text: hourlyForecast[index].time
                                            font.pixelSize: 10; font.family: Theme.fontFamily
                                            color: index === 0 ? Theme.primary : Theme.textDim
                                        }

                                        Canvas {
                                            Layout.alignment: Qt.AlignHCenter
                                            width: 24; height: 24
                                            onPaint: {
                                                var ctx = getContext("2d")
                                                ctx.clearRect(0, 0, width, height)
                                                drawWeatherIcon(ctx, hourlyForecast[index].icon, 12, 12, 24)
                                            }
                                            Component.onCompleted: requestPaint()
                                        }

                                        Text {
                                            Layout.alignment: Qt.AlignHCenter
                                            text: hourlyForecast[index].temp + "\u00B0"
                                            font.pixelSize: 13; font.weight: Font.DemiBold
                                            font.family: Theme.fontFamily
                                            color: index === 0 ? Theme.primary : Theme.text
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: 8 }

            /* ---- Bottom section: weekly forecast + details ---- */
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                Layout.leftMargin: 12; Layout.rightMargin: 12
                spacing: 10

                /* 7-Day Forecast */
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    color: Theme.surface; radius: Theme.radiusSmall

                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 12; spacing: 2

                        Text {
                            text: "7-Day Forecast"
                            font.pixelSize: 12; font.weight: Font.DemiBold
                            font.family: Theme.fontFamily; color: Theme.text
                        }

                        Repeater {
                            model: weeklyForecast.length

                            Rectangle {
                                Layout.fillWidth: true; Layout.fillHeight: true
                                color: weekDayMa.containsMouse ? Theme.hoverBg : "transparent"
                                radius: Theme.radiusSmall

                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: 4; anchors.rightMargin: 4

                                    Text {
                                        Layout.preferredWidth: 80
                                        text: weeklyForecast[index].day
                                        font.pixelSize: 12; font.family: Theme.fontFamily
                                        color: index === 0 ? Theme.primary : Theme.text
                                    }

                                    Canvas {
                                        width: 20; height: 20
                                        onPaint: {
                                            var ctx = getContext("2d")
                                            ctx.clearRect(0, 0, width, height)
                                            drawWeatherIcon(ctx, weeklyForecast[index].icon, 10, 10, 20)
                                        }
                                        Component.onCompleted: requestPaint()
                                    }

                                    Item { Layout.fillWidth: true }

                                    Text {
                                        text: weeklyForecast[index].high + "\u00B0"
                                        font.pixelSize: 13; font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily; color: Theme.text
                                    }

                                    Rectangle {
                                        Layout.preferredWidth: 40; height: 3; radius: 2
                                        color: Theme.surfaceLight

                                        Rectangle {
                                            x: parent.width * ((weeklyForecast[index].low - 30) / 50)
                                            width: parent.width * ((weeklyForecast[index].high - weeklyForecast[index].low) / 50)
                                            height: parent.height; radius: 2
                                            color: weeklyForecast[index].high > 70 ? Theme.warning
                                                 : weeklyForecast[index].high < 55 ? Theme.accent
                                                 : Theme.primary
                                        }
                                    }

                                    Text {
                                        text: weeklyForecast[index].low + "\u00B0"
                                        font.pixelSize: 12; font.family: Theme.fontFamily
                                        color: Theme.textDim
                                    }
                                }

                                MouseArea {
                                    id: weekDayMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }
                    }
                }

                /* Details Grid */
                Rectangle {
                    Layout.preferredWidth: 200; Layout.fillHeight: true
                    color: Theme.surface; radius: Theme.radiusSmall

                    GridLayout {
                        anchors.fill: parent; anchors.margins: 12
                        columns: 2; rowSpacing: 8; columnSpacing: 8

                        Text {
                            Layout.columnSpan: 2; text: "Details"
                            font.pixelSize: 12; font.weight: Font.DemiBold
                            font.family: Theme.fontFamily; color: Theme.text
                        }

                        Repeater {
                            model: [
                                { label: "Humidity",    value: humidity + "%",        icon: "droplet" },
                                { label: "Wind",        value: windSpeed + " mph",    icon: "wind" },
                                { label: "UV Index",    value: uvIndex.toString(),    icon: "sun" },
                                { label: "Pressure",    value: pressure + " hPa",     icon: "thermometer" },
                                { label: "Visibility",  value: visibility + " mi",    icon: "eye" },
                                { label: "Feels Like",  value: feelsLike + "\u00B0F", icon: "thermometer" }
                            ]

                            Rectangle {
                                Layout.fillWidth: true; Layout.fillHeight: true
                                color: Theme.surfaceAlt; radius: Theme.radiusSmall

                                ColumnLayout {
                                    anchors.centerIn: parent; spacing: 3

                                    RowLayout {
                                        Layout.alignment: Qt.AlignHCenter; spacing: 4
                                        Components.CanvasIcon {
                                            iconName: modelData.icon; iconSize: 12
                                            iconColor: Theme.textMuted
                                        }
                                        Text {
                                            text: modelData.label
                                            font.pixelSize: 10; font.family: Theme.fontFamily
                                            color: Theme.textMuted
                                        }
                                    }

                                    Text {
                                        Layout.alignment: Qt.AlignHCenter
                                        text: modelData.value
                                        font.pixelSize: 15; font.weight: Font.DemiBold
                                        font.family: Theme.fontFamily; color: Theme.text
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: 12 }
        }
    }
}
