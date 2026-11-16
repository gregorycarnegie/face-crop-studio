pragma Singleton
import QtQuick
import "fixture.js" as Fixture

// Stands in for the Rust Studio in the QML tests: every property, signal and method the QML
// uses, with the real factory settings, presets and shapes (fixture.js, generated from a fresh
// Studio). Tests set the state and read `calls` to see what the UI asked for.
QtObject {
    property bool ready: true
    property string engineLabel: "cpu · mock"
    property string engineError: ""
    property bool switching: false
    property string cudaStatus: ""
    property bool cudaAvailable: false
    property string cudaWarning: ""

    property int pending: 0
    property var queue: []
    property int current: -1
    property var currentRow: null
    property int selectedTotal: 0
    property int selectedPhotos: 0

    property string imageUrl: ""
    property int imageWidth: 0
    property int imageHeight: 0
    property var faces: []
    property var previews: []
    property bool canUndo: false
    property bool canRedo: false
    property real tilt: 0

    property var settings: Fixture.data.settings
    property var presets: Fixture.data.presets
    property var shapes: Fixture.data.shapes
    property string imageFilter: Fixture.data.imageFilter
    property string version: Fixture.data.version
    property string qtVersion: "6"
    property var shapeOutline: []
    property string presetName: ""

    property bool exporting: false
    property real exportProgress: 0
    property bool hasReport: false
    property var log: []
    property var mapping: ({ "loaded": false })

    property bool cameraOpen: false
    property bool cameraLive: false
    property string cameraStatus: ""
    property string cameraUrl: ""
    property var cameraFaces: []
    property var cameras: []

    signal queueRowChanged(int row, var item)
    signal queueRowRemoved(int row)
    signal imageChanged()
    signal toast(string message, string url)

    // Every method call, oldest first, as { name, args }.
    property var calls: []
    function record(name, args) {
        calls = calls.concat([{ "name": name, "args": Array.prototype.slice.call(args) }])
    }
    function called(name) {
        return calls.filter(c => c.name === name)
    }

    function addFace() { record("addFace", arguments) }
    function addFiles() { record("addFiles", arguments) }
    function clearMapping() { record("clearMapping", arguments) }
    function clearQueue() { record("clearQueue", arguments) }
    function closeCamera() { record("closeCamera", arguments) }
    function conflicts() { record("conflicts", arguments); return 0 }
    function deleteFace() { record("deleteFace", arguments) }
    function exportTo() { record("exportTo", arguments) }
    function loadMapping() { record("loadMapping", arguments) }
    function openCamera() { record("openCamera", arguments) }
    function paste() { record("paste", arguments) }
    function redo() { record("redo", arguments) }
    function remove() { record("remove", arguments) }
    function resetFace() { record("resetFace", arguments) }
    function restoreDefaults() { record("restoreDefaults", arguments) }
    function restoreEnhance() { record("restoreEnhance", arguments) }
    function rotate() { record("rotate", arguments) }
    function saveQueueList() { record("saveQueueList", arguments) }
    function saveReport() { record("saveReport", arguments) }
    function select() { record("select", arguments) }
    function selectAllFaces() { record("selectAllFaces", arguments) }
    function setCameraLive() { record("setCameraLive", arguments) }
    function setFaceBox() { record("setFaceBox", arguments) }
    function setMappingColumns() { record("setMappingColumns", arguments) }
    function setMappingSheet() { record("setMappingSheet", arguments) }
    function setMappingTable() { record("setMappingTable", arguments) }
    function setPreset() { record("setPreset", arguments) }
    function setTilt() { record("setTilt", arguments) }
    function snapshot() { record("snapshot", arguments) }
    function start() { record("start", arguments) }
    function toggleFace() { record("toggleFace", arguments) }
    function undo() { record("undo", arguments) }
    function update() { record("update", arguments) }
}
