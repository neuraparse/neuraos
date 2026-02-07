pragma Singleton
import QtQuick 2.15

QtObject {
    /* Master theme switch */
    property bool darkMode: Settings.theme !== "light"

    /* ─── Surfaces (deep blue-tinted darks, 2026 glassmorphism base) ─── */
    property color background:   darkMode ? "#0D0D12" : "#F0F1F6"
    property color surface:      darkMode ? "#161620" : "#FFFFFF"
    property color surfaceAlt:   darkMode ? "#1E1E2A" : "#F5F5FA"
    property color surfaceLight: darkMode ? "#2A2A38" : "#E5E6EE"

    /* ─── Glass (translucent panels) ─── */
    property color glass:       darkMode ? Qt.rgba(0.10, 0.10, 0.15, 0.78) : Qt.rgba(1, 1, 1, 0.72)
    property color glassBorder: darkMode ? Qt.rgba(1, 1, 1, 0.08) : Qt.rgba(0, 0, 0, 0.06)
    property color glassHover:  darkMode ? Qt.rgba(1, 1, 1, 0.06) : Qt.rgba(0, 0, 0, 0.04)
    property color glassActive: darkMode ? Qt.rgba(1, 1, 1, 0.10) : Qt.rgba(0, 0, 0, 0.07)

    /* ─── Brand Accent (brighter, more vibrant) ─── */
    readonly property color primary:      "#5B9AFF"
    readonly property color primaryDim:   "#3D7CE6"
    readonly property color secondary:    "#A78BFA"
    readonly property color secondaryDim: "#8B6FE0"
    readonly property color accent:       "#38BDF8"

    /* ─── Status ─── */
    readonly property color success: "#34D399"
    readonly property color warning: "#FBBF24"
    readonly property color error:   "#F87171"

    /* ─── Text (better contrast) ─── */
    property color text:      darkMode ? "#EAEDF3" : "#141420"
    property color textDim:   darkMode ? "#8B8FA2" : "#5C6075"
    property color textMuted: darkMode ? "#505468" : "#9298A8"

    /* ─── Sizing (larger radii, 2026 trend) ─── */
    readonly property int radius:      16
    readonly property int radiusSmall: 10
    readonly property int radiusTiny:  6
    readonly property int panelHeight:  48
    readonly property int dockHeight:   68
    readonly property int cardPadding:  16
    readonly property int spacing:      12

    /* ─── Typography ─── */
    readonly property string fontFamily: "Inter"
    readonly property int fontSizeSmall:  11
    readonly property int fontSizeNormal: 13
    readonly property int fontSizeLarge:  16
    readonly property int fontSizeXL:     22
    readonly property int fontSizeHuge:   48

    /* ─── Animation (smoother, spring-like) ─── */
    readonly property int animFast:   180
    readonly property int animNormal: 300
    readonly property int animSlow:   450

    /* ─── Window Chrome ─── */
    property color windowBg:              darkMode ? "#161620" : "#FFFFFF"
    property color windowTitleBar:        darkMode ? "#181822" : "#F2F3F7"
    property color windowTitleBarFocused: darkMode ? "#1E1E2A" : "#EAEBF2"
    property color windowBorder:          darkMode ? "#2A2A36" : "#D8DAE4"
    property color windowBorderFocused:   darkMode ? "#3A3A4A" : "#C0C3D0"
    readonly property int windowRadius: 16
    readonly property int windowTitleH: 42
    readonly property int windowShadow: 24
    readonly property int windowMinW:   400
    readonly property int windowMinH:   300

    /* ─── Shadow ─── */
    property color shadowColor: darkMode ? Qt.rgba(0, 0, 0, 0.55) : Qt.rgba(0, 0, 0, 0.12)

    /* ─── Elevation ─── */
    property color elevated:       darkMode ? "#1C1C28" : "#FFFFFF"
    property color elevatedBorder: darkMode ? "#2A2A38" : "#E0E2EA"

    /* ─── Taskbar (floating dock) ─── */
    readonly property int taskbarH: 56
    property color taskbarBg:     darkMode ? Qt.rgba(0.08, 0.08, 0.11, 0.88) : Qt.rgba(0.96, 0.96, 0.97, 0.88)
    property color taskbarHover:  darkMode ? Qt.rgba(1, 1, 1, 0.08) : Qt.rgba(0, 0, 0, 0.05)
    property color taskbarActive: darkMode ? Qt.rgba(0.36, 0.60, 1.0, 0.15) : Qt.rgba(0.36, 0.60, 1.0, 0.18)

    /* ─── Start Menu ─── */
    property color startMenuBg: darkMode ? Qt.rgba(0.07, 0.07, 0.10, 0.92) : Qt.rgba(0.97, 0.97, 0.98, 0.92)
    readonly property int startMenuW: 580
    readonly property int startMenuH: 700

    /* ─── Desktop ─── */
    property color desktopWidgetBg: darkMode ? Qt.rgba(0.10, 0.10, 0.14, 0.80) : Qt.rgba(1, 1, 1, 0.82)
    readonly property int desktopWidgetRadius: 14

    /* ─── Notification Center ─── */
    readonly property int notifCenterW: 380
    property color notifCenterBg: darkMode ? Qt.rgba(0.07, 0.07, 0.10, 0.92) : Qt.rgba(0.97, 0.97, 0.98, 0.92)

    /* ─── Accent Glows ─── */
    property color glowPrimary:   darkMode ? Qt.rgba(0.36, 0.60, 1.0, 0.12) : Qt.rgba(0.36, 0.60, 1.0, 0.06)
    property color glowSecondary: darkMode ? Qt.rgba(0.65, 0.55, 0.98, 0.10) : Qt.rgba(0.65, 0.55, 0.98, 0.05)

    /* ─── Hover / Active helpers ─── */
    property color hoverBg:  darkMode ? Qt.rgba(1, 1, 1, 0.06) : Qt.rgba(0, 0, 0, 0.04)
    property color activeBg: darkMode ? Qt.rgba(1, 1, 1, 0.10) : Qt.rgba(0, 0, 0, 0.07)

    /* ─── AI Bus & Orchestration (CosmOS-inspired) ─── */
    readonly property color aiBus:      "#06B6D4"
    readonly property color aiBusDim:   "#0891B2"

    /* ─── AI Memory (Steve OS-inspired) ─── */
    readonly property color aiMemory:    "#A78BFA"
    readonly property color aiMemoryDim: "#8B6FE0"

    /* ─── Automation (WarmWind OS-inspired) ─── */
    readonly property color automation:    "#F59E0B"
    readonly property color automationDim: "#D97706"

    /* ─── Command Palette (Bytebot-inspired) ─── */
    property color commandPaletteBg: darkMode ? Qt.rgba(0.06, 0.06, 0.09, 0.95) : Qt.rgba(0.97, 0.97, 0.98, 0.95)
    readonly property int commandPaletteW: 620
    readonly property int commandPaletteH: 420

    /* ─── MCP & Knowledge (Archon OS-inspired) ─── */
    readonly property color mcp:         "#10B981"
    readonly property color mcpDim:      "#059669"
    readonly property color knowledge:   "#3B82F6"
    readonly property color knowledgeDim:"#2563EB"

    /* ─── Ecosystem (Fuchsia/CosmOS-inspired) ─── */
    readonly property color ecosystem:    "#EC4899"
    readonly property color ecosystemDim: "#DB2777"
}
