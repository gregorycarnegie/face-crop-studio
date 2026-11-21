pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

Rectangle {
    id: root

    color: Theme.bg1

    // Mirrors Studio.queue in place, so the list (and its scroll position) is not reset when
    // photos are added. A single photo's change or removal comes as its own signal.
    function sync() {
        const items = Studio.queue
        if (items.length === 0) {
            queueModel.clear()
            return
        }
        for (let i = 0; i < items.length; ++i) {
            if (i < queueModel.count)
                queueModel.set(i, items[i])
            else
                queueModel.append(items[i])
        }
        while (queueModel.count > items.length)
            queueModel.remove(queueModel.count - 1)
    }

    ListModel { id: queueModel }

    Connections {
        target: Studio
        function onQueueChanged() { root.sync() }
        function onQueueRowChanged(row, item) { queueModel.set(row, item) }
        function onQueueRowRemoved(row) { queueModel.remove(row) }
    }

    Rectangle {
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: 1
        color: Theme.rule
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.rightMargin: 1
        spacing: 0

        Segmented {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            Layout.bottomMargin: 6
            value: UiState.sidebarTab
            options: [
                { value: "photos", label: queueModel.count > 0 ? qsTr("Photos · %1").arg(queueModel.count) : qsTr("Photos") },
                { value: "mapping", label: qsTr("Mapping") },
                { value: "activity", label: qsTr("Activity") }
            ]
            onActivated: value => UiState.sidebarTab = value
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: ["photos", "mapping", "activity"].indexOf(UiState.sidebarTab)

            ColumnLayout {
                spacing: 0

                RowLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: Theme.pad
                    Layout.rightMargin: Theme.pad - 6
                    Layout.bottomMargin: 4
                    Layout.preferredHeight: 32
                    spacing: 8

                    Spinner {
                        visible: Studio.pending > 0
                        size: 12
                    }
                    Text {
                        Layout.fillWidth: true
                        text: Studio.pending > 0 ? qsTr("Finding faces · %1 to go").arg(Studio.pending)
                            : queueModel.count > 0 ? qsTr("All photos ready") : ""
                        color: Theme.ink3
                        font.pixelSize: Theme.caption
                        elide: Text.ElideRight
                        // A plain Text reaches screen readers with no name; this is the only
                        // place detection progress is shown.
                        Accessible.role: Accessible.StaticText
                        Accessible.name: text
                    }
                    AppButton {
                        visible: queueModel.count > 0
                        implicitHeight: 28
                        leftPadding: 10
                        rightPadding: 10
                        variant: "ghost"
                        text: qsTr("Clear")
                        font.pixelSize: Theme.caption
                        onClicked: clearDialog.open()
                    }
                    IconButton {
                        id: moreButton
                        visible: queueModel.count > 0 || Studio.hasReport
                        side: 28
                        iconPath: Icons.more
                        tip: qsTr("Save the queue list or the last export report")
                        onClicked: moreMenu.popup(moreButton, 0, moreButton.height + 4)
                    }
                }

                ListView {
                    id: list

                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: queueModel
                    spacing: 2
                    leftMargin: 8
                    rightMargin: 8
                    bottomMargin: 8
                    currentIndex: Studio.current
                    activeFocusOnTab: true
                    boundsBehavior: Flickable.StopAtBounds
                    highlightMoveDuration: Theme.medium
                    Accessible.role: Accessible.List
                    Accessible.name: qsTr("Photos")
                    Keys.onUpPressed: Studio.select(Studio.current - 1)
                    Keys.onDownPressed: Studio.select(Studio.current + 1)
                    Keys.onDeletePressed: Studio.remove(Studio.current)

                    ScrollBar.vertical: ScrollBar {}

                    add: Transition {
                        OpacityAnimator { from: 0; to: 1; duration: Theme.medium; easing.type: Easing.OutCubic }
                    }

                    delegate: QueueDelegate {
                        width: ListView.view.width - ListView.view.leftMargin - ListView.view.rightMargin
                        onClicked: Studio.select(index)
                        onRemoveRequested: Studio.remove(index)
                    }

                    Text {
                        anchors.centerIn: parent
                        width: parent.width - 48
                        visible: queueModel.count === 0
                        text: qsTr("Photos you add appear here.")
                        horizontalAlignment: Text.AlignHCenter
                        wrapMode: Text.Wrap
                        color: Theme.ink3
                        font.pixelSize: Theme.caption
                    }
                }
            }

            MappingTab {}
            ActivityLog {}
        }
    }

    AppMenu {
        id: moreMenu
        Action {
            text: qsTr("Save queue list…")
            enabled: queueModel.count > 0
            onTriggered: queueListDialog.open()
        }
        Action {
            text: qsTr("Save last export report…")
            enabled: Studio.hasReport && !Studio.exporting
            onTriggered: reportDialog.open()
        }
    }

    // Saves the `chosen` file URL as `kind` ("report" or "queue"), adding `suffix` (the chosen
    // filter's extension) to a bare name. The dialogs have no defaultSuffix: Qt's GTK dialog adds
    // it only after GTK's overwrite check, so a bare name replaced the suffixed file unasked, and
    // the suffix ignored the filter (JSON saved as .csv). A file only the added extension names
    // is asked about here; the dialog itself asks about a name typed in full.
    function saveAs(chosen: string, suffix: string, kind: string) {
        const url = Studio.withSuffix(chosen, suffix)
        if (url !== chosen && Studio.fileExists(url)) {
            replaceDialog.url = url
            replaceDialog.kind = kind
            replaceDialog.open()
        } else {
            save(url, kind)
        }
    }
    function save(url: string, kind: string) {
        if (kind === "report")
            Studio.saveReport(url)
        else
            Studio.saveQueueList(url)
    }

    // No preset file name: Qt rejects a name that doesn't exist yet (a startup warning when
    // declared), and Windows' native dialog ignores one set at open time.
    FileDialog {
        id: queueListDialog
        title: qsTr("Save queue list")
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("Text files (*.txt)")]
        onAccepted: root.saveAs(selectedFile.toString(), "txt", "queue")
    }

    // JSON for a .json name (with success and failure totals), CSV otherwise.
    FileDialog {
        id: reportDialog
        title: qsTr("Save export report")
        fileMode: FileDialog.SaveFile
        nameFilters: [qsTr("CSV (*.csv)"), qsTr("JSON (*.json)")]
        onAccepted: root.saveAs(selectedFile.toString(), selectedNameFilter.extensions[0] ?? "csv", "report")
    }

    ChoiceDialog {
        id: replaceDialog

        property string url
        property string kind

        title: qsTr("Replace “%1”?").arg(decodeURIComponent(url.substring(url.lastIndexOf("/") + 1)))
        message: qsTr("A file with that name is already in the folder. Replacing it overwrites its contents.")
        choices: [
            { value: "replace", label: qsTr("Replace"), variant: "primary" },
            { value: "cancel", label: qsTr("Cancel"), variant: "ghost" }
        ]
        onChosen: value => {
            if (value === "replace")
                root.save(url, kind)
        }
    }

    ChoiceDialog {
        id: clearDialog
        title: qsTr("Clear all photos?")
        message: qsTr("This removes %1 from the queue, with any face edits. The files themselves stay where they are.")
            .arg(queueModel.count === 1 ? qsTr("1 photo") : qsTr("%1 photos").arg(queueModel.count))
        choices: [
            { value: "clear", label: qsTr("Clear"), variant: "primary" },
            { value: "cancel", label: qsTr("Cancel"), variant: "ghost" }
        ]
        onChosen: value => { if (value === "clear") Studio.clearQueue() }
    }
}
