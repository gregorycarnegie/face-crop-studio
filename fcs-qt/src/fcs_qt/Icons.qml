pragma Singleton
import QtQuick

// Stroke icons on a 24-unit grid, drawn by Icon.qml.
QtObject {
    readonly property string plus: "M12 5v14M5 12h14"
    readonly property string check: "M5 12.5l4.5 4.5L19 7"
    readonly property string alert: "M12 9v4.5M12 17h.01M10.3 4.2L2.6 18a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z"
    readonly property string info: "M12 21a9 9 0 1 0 0-18a9 9 0 0 0 0 18zM12 11v5M12 8h.01"
    readonly property string upload: "M12 15V4M7.5 8.5L12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
    readonly property string folder: "M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v7.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z"
    readonly property string faceScan: "M4 8.5V6.5A2.5 2.5 0 0 1 6.5 4H8.5M15.5 4h2A2.5 2.5 0 0 1 20 6.5v2M20 15.5v2a2.5 2.5 0 0 1-2.5 2.5h-2M8.5 20h-2A2.5 2.5 0 0 1 4 17.5v-2M9.5 10v.5M14.5 10v.5M9.5 15a3.5 3.5 0 0 0 5 0"
}
