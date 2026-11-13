import QtQuick

// Photos or folders dropped anywhere on the window join the queue.
DropArea {
    id: root

    keys: ["text/uri-list"]
    onDropped: drop => {
        Studio.addFiles(drop.urls.map(url => url.toString()))
        drop.acceptProposedAction()
    }

    Rectangle {
        anchors.fill: parent
        anchors.margins: 10
        radius: 16
        color: Qt.alpha(Theme.accent, 0.07)
        border.width: 2
        border.color: Theme.accent
        opacity: root.containsDrag ? 1 : 0
        visible: opacity > 0

        Behavior on opacity { OpacityAnimator { duration: Theme.fast } }

        Rectangle {
            anchors.centerIn: parent
            implicitWidth: dropLabel.implicitWidth + 36
            implicitHeight: 48
            radius: 24
            color: Theme.accent

            Text {
                id: dropLabel
                anchors.centerIn: parent
                text: qsTr("Drop to add photos")
                color: Theme.textOnAccent
                font.pixelSize: Theme.title
                font.weight: Font.DemiBold
            }
        }
    }
}
