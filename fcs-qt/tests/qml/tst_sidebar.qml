import QtQuick
import QtTest
import fcs_qt
import "helpers.js" as H

Item {
    width: 360
    height: 720

    Sidebar {
        id: sidebar
        anchors.fill: parent
    }

    TestCase {
        name: "Sidebar"
        when: windowShown

        function list() { return H.descendant(sidebar, i => i instanceof ListView) }
        function shows(text) { return H.descendant(sidebar, i => i.text === text) !== null }

        function init() {
            Studio.queue = []
            Studio.pending = 0
            Studio.current = -1
            Studio.calls = []
        }

        // The quadratic-batch fix: a whole queue arrives once, then single rows change alone.
        function test_rows_follow_the_queue_and_its_row_signals() {
            Studio.queue = [H.row("a.jpg"), H.row("b.jpg"), H.row("c.jpg")]
            compare(list().count, 3)
            Studio.queueRowChanged(1, H.row("b.jpg", { "faces": 4, "selected": 2 }))
            compare(list().model.get(1).faces, 4)
            compare(list().model.get(1).selected, 2)
            Studio.queueRowRemoved(0)
            compare(list().count, 2)
            compare(list().model.get(0).name, "b.jpg")
            Studio.queue = []
            compare(list().count, 0)
            verify(shows("Photos you add appear here."))
        }

        function test_the_status_line_reports_detection() {
            Studio.queue = [H.row("a.jpg", { "status": "queued" }), H.row("b.jpg")]
            Studio.pending = 1
            verify(shows("Finding faces · 1 to go"))
            Studio.pending = 0
            verify(shows("All photos ready"))
        }

        function test_keys_move_through_and_remove_photos() {
            Studio.queue = [H.row("a.jpg"), H.row("b.jpg")]
            Studio.current = 0
            list().forceActiveFocus()
            keyClick(Qt.Key_Down)
            compare(Studio.called("select")[0].args, [1])
            keyClick(Qt.Key_Delete)
            compare(Studio.called("remove")[0].args, [0])
        }

        // Qt's GTK save dialog checks for an existing file before adding a defaultSuffix, so the
        // sidebar adds the suffix itself and asks about a file only that suffix names.
        function test_a_bare_save_name_takes_the_suffix_and_asks_before_replacing() {
            const replace = sidebar.data.find(o => o instanceof ChoiceDialog && o.choices.length === 2
                                              && o.choices[0].value === "replace")
            const saved = () => Studio.called("saveReport").map(c => c.args[0])
            sidebar.saveAs("file:///d/new", "json", "report")
            compare(saved(), ["file:///d/new.json"])
            sidebar.saveAs("file:///d/list", "txt", "queue")
            compare(Studio.called("saveQueueList")[0].args, ["file:///d/list.txt"])

            Studio.existingFiles = ["file:///d/old.csv"]
            sidebar.saveAs("file:///d/old", "csv", "report")
            tryVerify(() => replace.opened)
            compare(saved().length, 1)
            replace.chosen("cancel")
            replace.close()
            compare(saved().length, 1)

            sidebar.saveAs("file:///d/old", "csv", "report")
            tryVerify(() => replace.opened)
            replace.chosen("replace")
            replace.close()
            compare(saved()[1], "file:///d/old.csv")

            // A name typed in full: the dialog has asked already.
            sidebar.saveAs("file:///d/old.csv", "csv", "report")
            compare(saved()[2], "file:///d/old.csv")
            Studio.existingFiles = []
            tryVerify(() => !replace.visible)
        }

        function test_clicking_a_row_selects_it() {
            Studio.queue = [H.row("a.jpg"), H.row("b.jpg")]
            tryVerify(() => list().itemAtIndex(1) !== null)
            mouseClick(list().itemAtIndex(1))
            compare(Studio.called("select")[0].args, [1])
        }
    }
}
