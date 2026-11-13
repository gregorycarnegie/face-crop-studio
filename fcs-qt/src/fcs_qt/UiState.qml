pragma Singleton
import QtQuick

// View state shared between panels that is not a crop setting.
QtObject {
    property bool showCropOverlay: true
    // "crop", "shape", "enhance" or "output"
    property string inspectorTab: "crop"
}
