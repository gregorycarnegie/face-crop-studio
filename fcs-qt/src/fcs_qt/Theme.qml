pragma Singleton
import QtQuick

// Charcoal surfaces and orange accent, the palette fcs-gui and the website share.
QtObject {
    readonly property color bg: "#0b0b10"
    readonly property color bg1: "#101017"
    readonly property color surface: "#181822"
    readonly property color surface2: "#262632"
    readonly property color rule: "#383846"
    readonly property color rule2: "#7c7c8f"
    readonly property color ink: "#f4f3f7"
    readonly property color ink2: "#c6c5d0"
    readonly property color ink3: "#abaaba"
    readonly property color accent: "#f56924"
    readonly property color accentHover: "#ff7b38"
    readonly property color accentDown: "#dc5a1a"
    readonly property color peach: "#ffa06b"
    // Dark ink on the accent: white on #f56924 is only 3.1:1.
    readonly property color textOnAccent: "#17100b"
    readonly property color focus: "#8fcdff"
    readonly property color green: "#6fd68a"
    readonly property color red: "#ff5c5c"

    // Modular scale, base 14, ratio 1.2 (minor third): a dense tool, not a reading surface.
    readonly property int caption: 12
    readonly property int body: 14
    readonly property int title: 17
    readonly property int display: 24
    // Empty elsewhere: the platform's default UI font.
    readonly property string fontFamily: Qt.platform.os === "windows" ? "Segoe UI Variable Text" : ""

    readonly property int radius: 10
    readonly property int radiusSmall: 7
    readonly property int pad: 16

    readonly property int fast: 120
    readonly property int medium: 220
}
