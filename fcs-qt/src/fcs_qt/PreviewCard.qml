import QtQuick

Rectangle {
    id: root

    // { index, url } from Studio.previews
    required property var modelData

    radius: Theme.radius
    color: Theme.surface
    border.width: 1
    border.color: Theme.rule

    // The output's full frame: a crop that runs past the photo's edge is transparent there,
    // and should read as part of the output rather than as a small picture.
    Rectangle {
        x: crop.x + (crop.width - crop.paintedWidth) / 2
        y: crop.y + (crop.height - crop.paintedHeight) / 2
        width: crop.paintedWidth
        height: crop.paintedHeight
        visible: crop.status === Image.Ready
        color: Theme.surface2
        border.width: 1
        border.color: Theme.rule
    }

    Image {
        id: crop
        anchors.fill: parent
        anchors.margins: 8
        anchors.bottomMargin: 28
        source: root.modelData.url
        sourceSize: Qt.size(480, 480)
        fillMode: Image.PreserveAspectFit
        asynchronous: true
        opacity: status === Image.Ready ? 1 : 0
        Accessible.role: Accessible.Graphic
        Accessible.name: label.text

        Behavior on opacity { OpacityAnimator { duration: Theme.medium; easing.type: Easing.OutCubic } }
    }

    Text {
        id: label
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: 10
        anchors.bottomMargin: 7
        text: qsTr("Face %1").arg(root.modelData.index + 1)
        color: Theme.ink3
        font.pixelSize: Theme.caption
    }
}
