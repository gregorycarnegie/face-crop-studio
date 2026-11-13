pragma ComponentBehavior: Bound
import QtQuick

Rectangle {
    id: root

    signal addPhotos

    readonly property var entry: Studio.current >= 0 && Studio.current < Studio.queue.length
                                 ? Studio.queue[Studio.current] : null
    readonly property bool hasImage: Studio.imageUrl !== ""

    color: Theme.bg

    EmptyState {
        anchors.centerIn: parent
        visible: root.entry === null
        onAddPhotos: root.addPhotos()
    }

    Column {
        anchors.centerIn: parent
        visible: root.entry !== null && !root.hasImage
        spacing: 14

        Spinner {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: root.entry?.status === "queued"
            size: 28
        }
        Icon {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: root.entry?.status === "failed"
            path: Icons.alert
            color: Theme.red
            size: 32
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: root.entry?.status === "failed" ? qsTr("Couldn't read %1").arg(root.entry.name)
                : Studio.ready ? qsTr("Finding faces…") : qsTr("Loading models…")
            color: Theme.ink2
            font.pixelSize: Theme.title
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.min(implicitWidth, root.width - 64)
            visible: root.entry?.status === "failed"
            text: root.entry?.error ?? ""
            wrapMode: Text.Wrap
            horizontalAlignment: Text.AlignHCenter
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
    }

    Image {
        id: photo

        anchors.fill: parent
        anchors.margins: 40
        anchors.bottomMargin: 84
        visible: root.hasImage
        source: Studio.imageUrl
        sourceSize: Qt.size(2560, 2560)
        fillMode: Image.PreserveAspectFit
        asynchronous: true
        opacity: status === Image.Ready ? 1 : 0
        Accessible.role: Accessible.Graphic
        Accessible.name: root.entry?.name ?? ""

        Behavior on opacity { OpacityAnimator { duration: Theme.medium; easing.type: Easing.OutCubic } }

        // The painted picture: face boxes are placed in fractions of it.
        Item {
            id: frame
            x: (photo.width - photo.paintedWidth) / 2
            y: (photo.height - photo.paintedHeight) / 2
            width: photo.paintedWidth
            height: photo.paintedHeight

            Rectangle {
                anchors.fill: parent
                anchors.margins: -1
                color: "transparent"
                border.width: 1
                border.color: Qt.alpha(Theme.ink, 0.08)
            }

            Repeater {
                model: Studio.faces
                delegate: FaceBox {
                    frameWidth: frame.width
                    frameHeight: frame.height
                }
            }
        }
    }

    FaceBar {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 22
        visible: root.hasImage
    }
}
