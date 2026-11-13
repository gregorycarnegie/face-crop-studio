pragma ComponentBehavior: Bound
import QtQuick

Rectangle {
    id: root

    signal addPhotos

    readonly property var entry: Studio.current >= 0 && Studio.current < Studio.queue.length
                                 ? Studio.queue[Studio.current] : null
    readonly property bool hasImage: Studio.imageUrl !== ""
    // The face whose handles show, by index into Studio.faces.
    property int activeFace: -1
    // Drag on the photo adds a face.
    property bool drawing: false
    // 1 fits the photo to the canvas.
    property real zoom: 1
    readonly property real maxZoom: 8

    function fit() {
        zoom = 1
        flick.contentX = 0
        flick.contentY = 0
    }
    function zoomBy(factor) {
        zoomAt(factor, flick.width / 2, flick.height / 2)
    }
    // Zoom by `factor`, keeping the photo point under (vx, vy) in the viewport where it is.
    function zoomAt(factor, vx, vy) {
        const next = Math.max(1, Math.min(maxZoom, zoom * factor))
        if (next === zoom)
            return
        const rx = (flick.contentX + vx - photo.x) / photo.width
        const ry = (flick.contentY + vy - photo.y) / photo.height
        zoom = next
        flick.contentX = Math.max(0, Math.min(flick.contentWidth - flick.width, photo.x + rx * photo.width - vx))
        flick.contentY = Math.max(0, Math.min(flick.contentHeight - flick.height, photo.y + ry * photo.height - vy))
    }

    color: Theme.bg
    // A crop larger than the photo draws past it; keep that out of the side panels.
    clip: true

    // A new photo starts fitted, with nothing active and no tool on. (Studio notifies
    // `current` through imageChanged, so a local copy gives it a change handler of its own.)
    readonly property int current: Studio.current
    onCurrentChanged: {
        activeFace = -1
        drawing = false
        fit()
    }
    Connections {
        target: Studio
        function onImageChanged() {
            if (root.activeFace >= Studio.faces.length)
                root.activeFace = -1
        }
    }

    Shortcut { sequence: "D"; enabled: root.hasImage; onActivated: root.drawing = !root.drawing }
    Shortcut { sequences: [StandardKey.ZoomIn, "Ctrl+="]; enabled: root.hasImage; onActivated: root.zoomBy(1.25) }
    Shortcut { sequences: [StandardKey.ZoomOut]; enabled: root.hasImage; onActivated: root.zoomBy(0.8) }
    Shortcut { sequence: "Ctrl+0"; enabled: root.hasImage; onActivated: root.fit() }
    Shortcut {
        sequences: [StandardKey.Delete, "Backspace"]
        enabled: root.activeFace >= 0
        onActivated: Studio.deleteFace(root.activeFace)
    }
    Shortcut {
        sequence: "Escape"
        enabled: root.drawing || root.activeFace >= 0
        onActivated: {
            root.drawing = false
            root.activeFace = -1
        }
    }

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
            visible: root.entry?.status !== "failed"
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
                : root.entry?.status === "done" ? qsTr("Opening photo…")
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

    // The photo's area: fitted inside it at zoom 1, larger and panned beyond.
    Item {
        id: stage

        readonly property real aspect: Studio.imageWidth / Math.max(1, Studio.imageHeight)
        readonly property real fitWidth: Math.min(width, height * aspect)
        readonly property real fitHeight: fitWidth / aspect

        anchors.fill: parent
        anchors.leftMargin: 72
        anchors.rightMargin: 40
        anchors.topMargin: 40
        anchors.bottomMargin: 84
        visible: root.hasImage

        Flickable {
            id: flick

            anchors.fill: parent
            contentWidth: page.width
            contentHeight: page.height
            // Drag pans only when there is somewhere to pan to; drawing owns drags.
            interactive: root.zoom > 1 && !root.drawing
            boundsBehavior: Flickable.StopAtBounds
            clip: root.zoom > 1

            Item {
                id: page
                width: Math.max(flick.width, photo.width)
                height: Math.max(flick.height, photo.height)

                // A click on bare canvas leaves the active face.
                TapHandler { onTapped: root.activeFace = -1 }

                Image {
                    id: photo

                    anchors.centerIn: parent
                    width: stage.fitWidth * root.zoom
                    height: stage.fitHeight * root.zoom
                    source: Studio.imageUrl
                    sourceSize: Qt.size(2048, 2048)
                    asynchronous: true
                    // Studio keeps the recent canvas copies itself; a second cached copy
                    // would only double the memory.
                    cache: false
                    smooth: true
                    mipmap: root.zoom < 1.5
                    opacity: status === Image.Ready ? 1 : 0
                    Accessible.role: Accessible.Graphic
                    Accessible.name: root.entry?.name ?? ""

                    Behavior on opacity { OpacityAnimator { duration: Theme.medium; easing.type: Easing.OutCubic } }

                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: -1
                        color: "transparent"
                        border.width: 1
                        border.color: Qt.alpha(Theme.ink, 0.08)
                    }

                    Repeater {
                        model: UiState.showCropOverlay ? Studio.faces : []
                        delegate: CropOverlay {
                            frameWidth: photo.width
                            frameHeight: photo.height
                        }
                    }

                    Repeater {
                        model: Studio.faces
                        delegate: FaceBox {
                            frameWidth: photo.width
                            frameHeight: photo.height
                            active: root.activeFace === modelData.index
                            editable: !root.drawing
                            onActivate: root.activeFace = modelData.index
                        }
                    }

                    // Draw tool: drag out a new face.
                    Item {
                        id: drawLayer

                        anchors.fill: parent
                        visible: root.drawing
                        z: 3

                        HoverHandler { cursorShape: Qt.CrossCursor }
                        DragHandler {
                            id: draw
                            target: null
                            grabPermissions: PointerHandler.CanTakeOverFromAnything
                            onActiveChanged: {
                                if (active || draft.width < 6 || draft.height < 6)
                                    return
                                Studio.addFace(draft.x / photo.width, draft.y / photo.height,
                                               draft.width / photo.width, draft.height / photo.height)
                                root.drawing = false
                                root.activeFace = Studio.faces.length - 1
                            }
                        }
                        Rectangle {
                            id: draft
                            readonly property point start: draw.centroid.pressPosition
                            readonly property point end: draw.centroid.position
                            visible: draw.active
                            x: Math.max(0, Math.min(start.x, end.x))
                            y: Math.max(0, Math.min(start.y, end.y))
                            width: Math.min(photo.width, Math.max(start.x, end.x)) - x
                            height: Math.min(photo.height, Math.max(start.y, end.y)) - y
                            radius: 6
                            color: Qt.alpha(Theme.accent, 0.12)
                            border.width: 2
                            border.color: Theme.accent
                        }
                    }
                }
            }
        }

        WheelHandler {
            target: null
            onWheel: event => {
                const p = point.position
                root.zoomAt(event.angleDelta.y > 0 ? 1.2 : 1 / 1.2, p.x, p.y)
            }
        }
    }

    CanvasTools {
        anchors.left: parent.left
        anchors.leftMargin: 14
        anchors.top: parent.top
        anchors.topMargin: 40
        visible: root.hasImage
        drawing: root.drawing
        zoom: root.zoom
        maxZoom: root.maxZoom
        onDrawToggled: root.drawing = !root.drawing
        onZoomRequested: factor => root.zoomBy(factor)
        onFitRequested: root.fit()
    }

    // What the draw tool does, while it is on.
    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 8
        visible: root.drawing
        implicitWidth: hint.implicitWidth + 24
        implicitHeight: 28
        radius: 14
        color: Theme.accent

        Text {
            id: hint
            anchors.centerIn: parent
            text: qsTr("Drag over a face to add it · Esc to cancel")
            color: Theme.textOnAccent
            font.pixelSize: Theme.caption
            font.weight: Font.DemiBold
        }
    }

    FaceBar {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 22
        visible: root.hasImage
    }
}
