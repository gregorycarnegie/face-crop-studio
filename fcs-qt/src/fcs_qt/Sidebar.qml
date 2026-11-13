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

        RowLayout {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            Layout.bottomMargin: 10
            spacing: 8

            Text {
                text: qsTr("Photos")
                color: Theme.ink3
                font.pixelSize: Theme.caption
                font.weight: Font.DemiBold
                font.letterSpacing: 0.9
                font.capitalization: Font.AllUppercase
                Accessible.role: Accessible.Heading
            }
            Text {
                visible: queueModel.count > 0
                text: queueModel.count
                color: Theme.ink2
                font.pixelSize: Theme.caption
            }
            Item { Layout.fillWidth: true }
            Spinner {
                visible: Studio.pending > 0
                size: 12
            }
            Text {
                visible: Studio.pending > 0
                text: qsTr("%1 to go").arg(Studio.pending)
                color: Theme.ink3
                font.pixelSize: Theme.caption
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

            ScrollBar.vertical: ScrollBar {}

            add: Transition {
                OpacityAnimator { from: 0; to: 1; duration: Theme.medium; easing.type: Easing.OutCubic }
            }

            delegate: QueueDelegate {
                width: ListView.view.width - ListView.view.leftMargin - ListView.view.rightMargin
                onClicked: Studio.select(index)
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
}
