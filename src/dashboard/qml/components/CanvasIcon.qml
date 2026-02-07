import QtQuick 2.15
import ".."

Item {
    id: icon
    width: iconSize; height: iconSize

    property string iconName: ""
    property color iconColor: Theme.text
    property int iconSize: 18

    FontLoader {
        id: remixFont
        source: "qrc:/fonts/remixicon.ttf"
    }

    Text {
        anchors.centerIn: parent
        text: getCodepoint(iconName)
        font.family: remixFont.name
        font.pixelSize: Math.round(iconSize * 0.9)
        color: iconColor
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        renderType: Text.NativeRendering
    }

    function getCodepoint(name) {
        var map = {
            /* App icons */
            "monitor":    "\uEA96",  /* bar-chart-2 */
            "terminal":   "\uF1F6",  /* terminal-box */
            "folder":     "\uED54",  /* folder-3 */
            "gear":       "\uF0E6",  /* settings-3 */
            "neural":     "\uEF30",  /* brain */
            "robot":      "\uF092",  /* robot */
            "chip":       "\uEBF0",  /* cpu */
            "drone":      "\uF005",  /* plane */
            "shield":     "\uF100",  /* shield-check */
            "atom":       "\uED3F",  /* flask */
            "wifi":       "\uF134",  /* wifi */
            "box":        "\uEA48",  /* archive */

            /* UI icons */
            "search":     "\uF0D1",  /* search */
            "bell":       "\uEF94",  /* notification-3 */
            "power":      "\uF126",  /* shut-down */
            "bluetooth":  "\uEACC",  /* bluetooth */
            "moon":       "\uEF75",  /* moon */
            "sun":        "\uF1BF",  /* sun */
            "info":       "\uEE59",  /* information */
            "file":       "\uED0F",  /* file-text */
            "code":       "\uEBAD",  /* code-s-slash */
            "image":      "\uEE4B",  /* image */
            "close":      "\uEB99",  /* close */
            "check":      "\uEB7B",  /* check */
            "plus":       "\uEA13",  /* add */
            "minus":      "\uF1AF",  /* subtract */
            "arrow-left": "\uEA64",  /* arrow-left-s */
            "arrow-right":"\uEA6E",  /* arrow-right-s */
            "grid":       "\uEDDF",  /* grid */
            "list":       "\uEEBE",  /* list-unordered */

            /* Additional icons */
            "user":       "\uF0E2",  /* user */
            "home":       "\uEE2B",  /* home */
            "star":       "\uF18B",  /* star */
            "heart":      "\uEE0F",  /* heart */
            "eye":        "\uECB5",  /* eye */
            "edit":       "\uEC86",  /* edit */
            "copy":       "\uECD5",  /* file-copy */
            "share":      "\uF0FE",  /* share */
            "link":       "\uEEB2",  /* link */
            "calendar":   "\uEB27",  /* calendar */
            "clock":      "\uF206",  /* time */
            "mail":       "\uEEF6",  /* mail */
            "camera":     "\uEB31",  /* camera */
            "mic":        "\uEF50",  /* mic */
            "volume":     "\uF0D8",  /* volume-up */
            "database":   "\uEC18",  /* database */
            "server":     "\uF0E0",  /* server */
            "cloud":      "\uEB9D",  /* cloud */
            "globe":      "\uEDCF",  /* global */
            "map-pin":    "\uEF14",  /* map-pin */
            "key":        "\uEE71",  /* key */
            "bug":        "\uEB07",  /* bug */
            "bookmark":   "\uEAE5",  /* bookmark */
            "flag":       "\uED3B",  /* flag */
            "tag":        "\uF025",  /* price-tag */
            "filter":     "\uED27",  /* filter */
            "sort":       "\uF15F",  /* sort-asc */
            "dashboard":  "\uEC14",  /* dashboard */
            "more":       "\uEF79",  /* more */
            "apps":       "\uEA44",  /* apps */
            "menu":       "\uEF3E",  /* menu */
            "download":   "\uEC5A",  /* download */
            "upload":     "\uED15",  /* upload */
            "lock":       "\uEECE",  /* lock */
            "refresh":    "\uF064",  /* refresh */
            "trash":      "\uEC2A",  /* delete-bin */
            "play":       "\uF00B",  /* play */
            "pause":      "\uEFD8",  /* pause */
            "stop":       "\uF1A1",  /* stop */
            "fullscreen": "\uED9C",  /* fullscreen */
            "layout":     "\uEE90",  /* layout-grid */
            "save":       "\uF0C5",  /* save */
            "zoom-in":    "\uF219",  /* zoom-in */
            "zoom-out":   "\uF21B",  /* zoom-out */
            "shuffle":    "\uF108",  /* shuffle */
            "repeat":     "\uF069",  /* repeat */
            "skip-back":  "\uF0BA",  /* skip-back */
            "skip-forward":"\uF0BD", /* skip-forward */
            "users":      "\uF21F",  /* group */

            /* v4.0 new icons */
            "send":       "\uF0D0",  /* send-plane */
            "attachment": "\uEA70",  /* attachment */
            "chat":       "\uEB51",  /* chat-3 */
            "ai-chat":    "\uEB51",  /* chat-3 (alias) */
            "arrow-up":   "\uEA78",  /* arrow-up-s */
            "arrow-down": "\uEA60",  /* arrow-down-s */
            "video":      "\uF115",  /* video */
            "alarm":      "\uEA21",  /* alarm */
            "timer":      "\uF206",  /* time (alias) */
            "stopwatch":  "\uF206",  /* time (alias) */
            "photo":      "\uEE4B",  /* image (alias) */
            "brightness": "\uF1BF",  /* sun (alias) */
            "airplane":   "\uF005",  /* plane (alias) */
            "sidebar":    "\uEE90",  /* layout-grid (alias) */
            "panel":      "\uEE90",  /* layout-grid (alias) */
            "weather":    "\uEB9D",  /* cloud (alias) */
            "thermometer":"\uF1F2",  /* temp-cold */
            "wind":       "\uF138",  /* windy */
            "droplet":    "\uEC5E",  /* drop */
            "snow":       "\uF14E",  /* snowy */
            "grid-view":  "\uEDDF",  /* grid (alias) */
            "list-view":  "\uEEBE",  /* list-unordered (alias) */
            "tab":        "\uF138",  /* window */
            "columns":    "\uEEB0",  /* layout-column */
            "pin":        "\uEFF2",  /* pushpin */
            "unpin":      "\uEFF2",  /* pushpin (alias) */
            "expand":     "\uED9C",  /* fullscreen (alias) */
            "collapse":   "\uEBCD",  /* contract */
            "chevron-right":"\uEA6E",/* arrow-right-s (alias) */
            "chevron-down":"\uEA60"  /* arrow-down-s (alias) */
        }
        return map[name] || "\uEE59"  /* fallback: info */
    }
}
