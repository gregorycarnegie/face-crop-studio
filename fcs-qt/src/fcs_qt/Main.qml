import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: window

    width: 1440
    height: 900
    minimumWidth: 1080
    minimumHeight: 680
    visible: true
    title: qsTr("Face Crop Studio")
    color: Theme.bg
    font.family: Theme.fontFamily
    font.pixelSize: Theme.body

    Component.onCompleted: {
        // The app is dark whatever the OS is set to; this also darkens the window frame.
        Qt.styleHints.colorScheme = Qt.ColorScheme.Dark
        Studio.start()
    }

    header: AppHeader {
        onAddPhotos: photoDialog.open()
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Sidebar {
            Layout.preferredWidth: 290
            Layout.fillHeight: true
        }
        CanvasView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            onAddPhotos: photoDialog.open()
        }
        Inspector {
            Layout.preferredWidth: 344
            Layout.fillHeight: true
            onExportRequested: folderDialog.open()
        }
    }

    DropOverlay {
        anchors.fill: parent
    }

    Toast {
        id: toast
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 28
    }

    Connections {
        target: Studio
        function onToast(message, url) { toast.show(message, url) }
    }

    Shortcut {
        sequences: [StandardKey.Open]
        onActivated: photoDialog.open()
    }
    // A focused text field takes Ctrl+V for itself first.
    Shortcut {
        sequences: [StandardKey.Paste]
        onActivated: Studio.paste()
    }
    Shortcut {
        sequence: "Ctrl+E"
        enabled: Studio.ready && Studio.selectedTotal > 0 && !Studio.exporting
        onActivated: folderDialog.open()
    }
    Shortcut {
        sequences: [StandardKey.SelectAll]
        onActivated: Studio.selectAllFaces(true)
    }
    Shortcut {
        sequences: [StandardKey.Undo]
        enabled: Studio.canUndo
        onActivated: Studio.undo()
    }
    Shortcut {
        sequences: [StandardKey.Redo, "Ctrl+Y"]
        enabled: Studio.canRedo
        onActivated: Studio.redo()
    }

    FileDialog {
        id: photoDialog
        title: qsTr("Add photos")
        fileMode: FileDialog.OpenFiles
        nameFilters: [Studio.imageFilter, qsTr("All files (*)")]
        onAccepted: Studio.addFiles(selectedFiles.map(url => url.toString()))
    }

    FolderDialog {
        id: folderDialog
        title: qsTr("Export faces to")
        onAccepted: {
            const folder = selectedFolder.toString()
            const clashes = Studio.conflicts(folder)
            if (clashes > 0) {
                conflictDialog.folder = folder
                conflictDialog.clashes = clashes
                conflictDialog.open()
            } else {
                Studio.exportTo(folder, "keep", UiState.report)
            }
        }
    }

    // Asked before an export would land on files already in the folder.
    ChoiceDialog {
        id: conflictDialog

        property string folder
        property int clashes

        title: qsTr("Some of these files already exist")
        message: clashes === 1
            ? qsTr("1 file in that folder has a name an exported face would take. Keep both saves the new crop as name(2); Replace overwrites it.")
            : qsTr("%1 files in that folder have names exported faces would take. Keep both saves new crops as name(2); Replace overwrites them.").arg(clashes)
        choices: [
            { value: "keep", label: qsTr("Keep both"), variant: "primary" },
            { value: "replace", label: qsTr("Replace") },
            { value: "cancel", label: qsTr("Cancel"), variant: "ghost" }
        ]
        onChosen: value => {
            if (value !== "cancel")
                Studio.exportTo(folder, value, UiState.report)
        }
    }
}
