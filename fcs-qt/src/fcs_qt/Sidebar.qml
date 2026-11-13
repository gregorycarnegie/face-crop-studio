pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    color: Theme.bg1

    // Mirrors Studio.queue in place, so a finished detection updates one row instead of
    // resetting the list (and its scroll position) mid-batch.
    function sync() {
        const items = Studio.queue
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
