import QtQuick
import QtTest
import fcs_qt
import "helpers.js" as H

Item {
    width: 420
    height: 900

    Inspector {
        id: inspector
        anchors.fill: parent
    }

    SignalSpy {
        id: exportSpy
        target: inspector
        signalName: "exportRequested"
    }

    TestCase {
        name: "Inspector"
        when: windowShown

        function clickable(test) {
            return H.descendant(inspector, i => typeof i.text === "string" && test(i.text)
                                && typeof i.clicked === "function")
        }

        function init() {
            Studio.ready = true
            Studio.exporting = false
            Studio.exportProgress = 0
            Studio.selectedTotal = 0
            Studio.selectedPhotos = 0
            Studio.calls = []
            exportSpy.clear()
        }

        function test_export_counts_the_selected_faces_and_asks_for_a_folder() {
            Studio.selectedTotal = 3
            Studio.selectedPhotos = 2
            verify(H.descendant(inspector, i => i.text === "Selected faces from 2 photos"))
            const go = clickable(t => t === "Export 3 faces")
            verify(go)
            mouseClick(go)
            compare(exportSpy.count, 1)
        }

        function test_a_running_export_shows_progress_and_ignores_clicks() {
            Studio.selectedTotal = 1
            Studio.exporting = true
            Studio.exportProgress = 0.5
            const going = clickable(t => t === "Exporting… 50%")
            verify(going)
            mouseClick(going)
            compare(exportSpy.count, 0)
        }

        function test_nothing_to_export_until_faces_are_selected() {
            const go = clickable(t => t === "Export 0 faces")
            verify(go)
            verify(!go.enabled)
        }

        function test_a_preset_card_applies_its_preset() {
            const card = H.descendant(inspector, i => i.modelData !== undefined && i.modelData !== null
                                      && i.modelData.name === "Passport" && typeof i.clicked === "function")
            verify(card, "the Passport preset card")
            mouseClick(card)
            compare(Studio.called("setPreset")[0].args, ["Passport"])
        }
    }
}
