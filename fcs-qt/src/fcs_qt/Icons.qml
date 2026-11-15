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
    readonly property string drawBox: "M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16M12 9v6M9 12h6"
    readonly property string zoomIn: "M11 18a7 7 0 1 0 0-14a7 7 0 0 0 0 14zM20 20l-4-4M11 8v6M8 11h6"
    readonly property string zoomOut: "M11 18a7 7 0 1 0 0-14a7 7 0 0 0 0 14zM20 20l-4-4M8 11h6"
    readonly property string fit: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"
    readonly property string rotateLeft: "M3 12a9 9 0 1 0 2.6-6.4L3 8M3 3v5h5"
    readonly property string rotateRight: "M21 12a9 9 0 1 1-2.6-6.4L21 8M21 3v5h-5"
    readonly property string straighten: "M3 18h18M5 15l14-7M12 5v2"
    readonly property string undo: "M9 14L4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11"
    readonly property string redo: "M15 14l5-5-5-5M20 9H9.5a5.5 5.5 0 0 0 0 11H13"
    readonly property string trash: "M4 7h16M10 11v6M14 11v6M5.5 7l1 12a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2l1-12M9 7V4h6v3"
    readonly property string restore: "M4 12a8 8 0 1 0 2.3-5.6L4 8.5M4 4v4.5h4.5M12 8v4l2.5 2.5"
    readonly property string more: "M4 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0M11 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0M18 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0"
    readonly property string close: "M6 6l12 12M18 6L6 18"
    readonly property string camera: "M4 8.5A2.5 2.5 0 0 1 6.5 6H8l1.5-2h5L16 6h1.5A2.5 2.5 0 0 1 20 8.5v8a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 16.5zM12 16a3.5 3.5 0 1 0 0-7a3.5 3.5 0 0 0 0 7z"
    readonly property string faceScan: "M4 8.5V6.5A2.5 2.5 0 0 1 6.5 4H8.5M15.5 4h2A2.5 2.5 0 0 1 20 6.5v2M20 15.5v2a2.5 2.5 0 0 1-2.5 2.5h-2M8.5 20h-2A2.5 2.5 0 0 1 4 17.5v-2M9.5 10v.5M14.5 10v.5M9.5 15a3.5 3.5 0 0 0 5 0"
}
