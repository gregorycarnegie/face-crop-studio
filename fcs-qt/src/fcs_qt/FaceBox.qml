pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls

// One face over the photo. Click to make it the active face; drag it, or the active face's
// handles, to move and resize it. The number chip (or Space) includes or skips it in export.
Item {
    id: root

    // { index, x, y, w, h, score, quality, selected, manual, modified } in photo fractions
    required property var modelData
    required property real frameWidth
    required property real frameHeight
    property bool active
    property bool editable: true
    signal activate

    readonly property bool selected: modelData.selected
    readonly property bool lit: hover.hovered || active
    readonly property real minSide: 12
    readonly property real baseX: modelData.x * frameWidth
    readonly property real baseY: modelData.y * frameHeight
    readonly property real baseW: modelData.w * frameWidth
    readonly property real baseH: modelData.h * frameHeight
    // Where an unfinished drag has the box, relative to where Studio has it.
    property real dx: 0
    property real dy: 0
    property real dw: 0
    property real dh: 0

    function commit() {
        Studio.setFaceBox(modelData.index, x / frameWidth, y / frameHeight,
                          width / frameWidth, height / frameHeight)
        dx = 0; dy = 0; dw = 0; dh = 0
    }
    // Drag an edge (hx, hy: 0 = left/top, 0.5 = untouched, 1 = right/bottom) by (ddx, ddy),
    // never past the opposite edge.
    function resize(hx, hy, ddx, ddy) {
        if (hx === 0) { const d = Math.min(ddx, baseW - minSide); dx = d; dw = -d }
        else if (hx === 1) dw = Math.max(ddx, minSide - baseW)
        if (hy === 0) { const d = Math.min(ddy, baseH - minSide); dy = d; dh = -d }
        else if (hy === 1) dh = Math.max(ddy, minSide - baseH)
    }

    x: baseX + dx
    y: baseY + dy
    width: baseW + dw
    height: baseH + dh
    z: active ? 2 : 1
    activeFocusOnTab: true
    Accessible.role: Accessible.CheckBox
    Accessible.name: qsTr("Face %1, %2 quality").arg(modelData.index + 1).arg(modelData.quality)
    Accessible.checked: selected
    Accessible.onToggleAction: Studio.toggleFace(modelData.index)
    Keys.onSpacePressed: Studio.toggleFace(modelData.index)
    Keys.onDeletePressed: Studio.deleteFace(modelData.index)
    onActiveFocusChanged: if (activeFocus) activate()

    HoverHandler {
        id: hover
        enabled: root.editable
        cursorShape: root.active ? Qt.SizeAllCursor : Qt.PointingHandCursor
    }
    TapHandler {
        enabled: root.editable
        // Grabs the press, so the canvas's "click on bare canvas" handler underneath does
        // not also see the tap and deactivate the face straight away.
        gesturePolicy: TapHandler.WithinBounds
        onTapped: root.activate()
    }
    DragHandler {
        id: move
        enabled: root.editable
        target: null
        grabPermissions: PointerHandler.CanTakeOverFromAnything
        onActiveChanged: active ? root.activate() : root.commit()
        onCentroidChanged: if (active) {
            root.dx = centroid.scenePosition.x - centroid.scenePressPosition.x
            root.dy = centroid.scenePosition.y - centroid.scenePressPosition.y
        }
    }

    // Not on the active face: its action bar sits where the tooltip would.
    ToolTip.visible: hover.hovered && !move.active && !root.active
    ToolTip.delay: 600
    ToolTip.text: root.modelData.manual
        ? qsTr("Face %1 · drawn by hand").arg(root.modelData.index + 1)
        : qsTr("Face %1 · sharpness %2 · %3% confident").arg(root.modelData.index + 1)
            .arg(root.modelData.quality.toLowerCase()).arg(Math.round(root.modelData.score * 100))

    Rectangle {
        anchors.fill: parent
        radius: 6
        color: root.selected ? Qt.alpha(Theme.accent, root.lit ? 0.16 : 0.07)
                             : Qt.alpha(Theme.ink, root.lit ? 0.10 : 0)
        border.width: root.active ? 2.5 : 2
        border.color: root.selected ? Theme.accent : Qt.alpha(Theme.ink, root.lit ? 0.95 : 0.55)

        Behavior on color { ColorAnimation { duration: Theme.fast } }
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

    // The number chip: click it to include or skip the face.
    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.top
        anchors.bottomMargin: 6
        implicitWidth: chip.implicitWidth + 14
        implicitHeight: 24
        radius: 7
        color: root.selected ? Theme.accent : Qt.alpha(Theme.bg, 0.88)
        border.width: root.selected ? 0 : 1
        border.color: Theme.rule

        Row {
            id: chip
            anchors.centerIn: parent
            spacing: 4

            Icon {
                anchors.verticalCenter: parent.verticalCenter
                path: root.selected ? Icons.check : Icons.close
                color: root.selected ? Theme.textOnAccent : Theme.ink3
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
        HoverHandler { cursorShape: Qt.PointingHandCursor }
        TapHandler {
            gesturePolicy: TapHandler.WithinBounds
            onTapped: Studio.toggleFace(root.modelData.index)
        }
    }

    // The active face's actions.
    Rectangle {
        visible: root.active && root.editable && !move.active
        anchors.right: parent.right
        anchors.bottom: parent.top
        anchors.bottomMargin: 4
        implicitWidth: actions.implicitWidth + 6
        implicitHeight: actions.implicitHeight + 6
        radius: Theme.radiusSmall + 2
        color: Theme.surface
        border.width: 1
        border.color: Theme.rule

        Row {
            id: actions
            anchors.centerIn: parent
            spacing: 2

            IconButton {
                side: 28
                iconPath: root.selected ? Icons.close : Icons.check
                tip: root.selected ? qsTr("Skip this face in export (Space)") : qsTr("Include this face in export (Space)")
                onClicked: Studio.toggleFace(root.modelData.index)
            }
            IconButton {
                side: 28
                visible: root.modelData.modified && !root.modelData.manual
                iconPath: Icons.restore
                tip: qsTr("Put the box back where the detector found it")
                onClicked: Studio.resetFace(root.modelData.index)
            }
            IconButton {
                side: 28
                iconPath: Icons.trash
                tip: qsTr("Remove this face (Delete)")
                onClicked: Studio.deleteFace(root.modelData.index)
            }
        }
    }

    // Resize handles on the corners and edge midpoints.
    Repeater {
        model: root.active && root.editable
               ? [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5]] : []

        delegate: Rectangle {
            id: handle

            required property var modelData
            readonly property real hx: modelData[0]
            readonly property real hy: modelData[1]

            x: hx * root.width - width / 2
            y: hy * root.height - height / 2
            width: 12
            height: 12
            radius: 3
            color: Theme.ink
            border.width: 2
            border.color: Theme.accent
            Accessible.ignored: true

            // A larger invisible grab area than the visible square.
            Item {
                anchors.fill: parent
                anchors.margins: -6

                HoverHandler {
                    cursorShape: handle.hx === 0.5 ? Qt.SizeVerCursor
                               : handle.hy === 0.5 ? Qt.SizeHorCursor
                               : (handle.hx === handle.hy) ? Qt.SizeFDiagCursor : Qt.SizeBDiagCursor
                }
                DragHandler {
                    target: null
                    grabPermissions: PointerHandler.CanTakeOverFromAnything
                    onActiveChanged: if (!active) root.commit()
                    onCentroidChanged: if (active) root.resize(handle.hx, handle.hy,
                        centroid.scenePosition.x - centroid.scenePressPosition.x,
                        centroid.scenePosition.y - centroid.scenePressPosition.y)
                }
            }
        }
    }
}
