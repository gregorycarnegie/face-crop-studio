pragma Singleton
import QtQuick

// View state shared between panels that is not a crop setting.
QtObject {
    property bool showCropOverlay: true
    // "crop", "shape", "enhance" or "output"
    property string inspectorTab: "crop"
    // "photos", "mapping" or "activity"
    property string sidebarTab: "photos"
    // Export report: "none", "csv" or "json"
    property string report: "none"
}
