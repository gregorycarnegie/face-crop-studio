import QtQuick
import QtQuick.Controls

// One detected face over the photo. Click (or Space) includes or excludes it from export.
Item {
    id: root

    // { index, x, y, w, h, score, quality, selected } with x..h in fractions of the photo
    required property var modelData
    required property real frameWidth
    required property real frameHeight
    readonly property bool selected: modelData.selected
    readonly property bool lit: hover.hovered || activeFocus

    x: modelData.x * frameWidth
    y: modelData.y * frameHeight
    width: modelData.w * frameWidth
    height: modelData.h * frameHeight
    activeFocusOnTab: true
    Accessible.role: Accessible.CheckBox
    Accessible.name: qsTr("Face %1, %2 quality").arg(modelData.index + 1).arg(modelData.quality)
    Accessible.checked: selected
    Accessible.onToggleAction: Studio.toggleFace(modelData.index)
    Keys.onSpacePressed: Studio.toggleFace(modelData.index)

    HoverHandler {
        id: hover
        cursorShape: Qt.PointingHandCursor
    }
    ToolTip.visible: hover.hovered
    ToolTip.delay: 500
    ToolTip.text: qsTr("Face %1 · sharpness %2 · %3% confident · click to %4")
        .arg(modelData.index + 1).arg(modelData.quality.toLowerCase())
        .arg(Math.round(modelData.score * 100)).arg(selected ? qsTr("skip") : qsTr("include"))
    TapHandler {
        onTapped: Studio.toggleFace(root.modelData.index)
    }

    Rectangle {
        anchors.fill: parent
        radius: 6
        color: root.selected ? Qt.alpha(Theme.accent, root.lit ? 0.18 : 0.08)
                             : Qt.alpha(Theme.ink, root.lit ? 0.10 : 0)
        border.width: 2
        border.color: root.selected ? Theme.accent : Qt.alpha(Theme.ink, root.lit ? 0.95 : 0.55)

        Behavior on color { ColorAnimation { duration: Theme.fast } }
        Behavior on border.color { ColorAnimation { duration: Theme.fast } }
    }

    Rectangle {
        anchors.fill: parent
        anchors.margins: -5
        radius: 10
        color: "transparent"
        border.width: 2
        border.color: Theme.focus
        visible: root.activeFocus
    }

    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.top
        anchors.bottomMargin: 6
        implicitWidth: chip.implicitWidth + 14
        implicitHeight: 22
        radius: 6
        color: root.selected ? Theme.accent : Qt.alpha(Theme.bg, 0.86)
        border.width: root.selected ? 0 : 1
        border.color: Theme.rule

        Row {
            id: chip
            anchors.centerIn: parent
            spacing: 4

            Icon {
                anchors.verticalCenter: parent.verticalCenter
                visible: root.selected
                path: Icons.check
                color: Theme.textOnAccent
                size: 13
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: root.modelData.index + 1
                color: root.selected ? Theme.textOnAccent : Theme.ink2
                font.pixelSize: Theme.caption
                font.weight: Font.DemiBold
            }
        }
    }
}
