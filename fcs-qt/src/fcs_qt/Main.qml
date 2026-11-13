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
    Shortcut {
        sequence: "Ctrl+E"
        enabled: Studio.ready && Studio.selectedTotal > 0 && !Studio.exporting
        onActivated: folderDialog.open()
    }
    Shortcut {
        sequences: [StandardKey.SelectAll]
        onActivated: Studio.selectAllFaces(true)
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
        onAccepted: Studio.exportTo(selectedFolder.toString())
    }
}
