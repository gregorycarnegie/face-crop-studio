import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects

// The canvas's tool rail: draw a face, zoom, turn the photo, undo.
Rectangle {
    id: root

    // The canvas's state, shown here; its actions go back out as signals.
    property bool drawing
    property real zoom: 1
    property real maxZoom: 8
    signal drawToggled
    signal zoomRequested(real factor)
    signal fitRequested

    implicitWidth: column.implicitWidth + 8
    implicitHeight: column.implicitHeight + 8
    radius: Theme.radius + 2
    color: Theme.surface
    border.width: 1
    border.color: Theme.rule
    layer.enabled: true
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowColor: "#80000000"
        shadowBlur: 0.7
        shadowVerticalOffset: 4
    }

    component Divider: Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        width: 20
        height: 1
        color: Theme.rule
    }

    Column {
        id: column
        anchors.centerIn: parent
        spacing: 2

        IconButton {
            iconPath: Icons.drawBox
            tip: qsTr("Draw a face the detector missed (D)")
            active: root.drawing
            onClicked: root.drawToggled()
        }
        Divider {}
        IconButton {
            iconPath: Icons.zoomIn
            tip: qsTr("Zoom in (Ctrl++, or the mouse wheel)")
            enabled: root.zoom < root.maxZoom
            onClicked: root.zoomRequested(1.25)
        }
        IconButton {
            iconPath: Icons.zoomOut
            tip: qsTr("Zoom out (Ctrl+-)")
            enabled: root.zoom > 1
            onClicked: root.zoomRequested(0.8)
        }
        IconButton {
            iconPath: Icons.fit
            tip: qsTr("Fit to window (Ctrl+0)")
            enabled: root.zoom > 1
            onClicked: root.fitRequested()
        }
        Divider {}
        IconButton {
            iconPath: Icons.rotateLeft
            tip: qsTr("Rotate left; finds the faces again")
            onClicked: Studio.rotate(-1)
        }
        IconButton {
            iconPath: Icons.rotateRight
            tip: qsTr("Rotate right; finds the faces again")
            onClicked: Studio.rotate(1)
        }
        IconButton {
            id: straightenButton
            iconPath: Icons.straighten
            tip: qsTr("Straighten")
            active: straighten.opened || Studio.tilt !== 0
            onClicked: straighten.opened ? straighten.close() : straighten.open()

            Popup {
                id: straighten
                x: parent.width + 10
                y: (parent.height - height) / 2
                width: 260
                padding: 14
                background: Rectangle {
                    radius: Theme.radius
                    color: Theme.surface
                    border.width: 1
                    border.color: Theme.rule
                }

                Column {
                    width: parent.width
                    spacing: 6

                    Row {
                        width: parent.width
                        Text {
                            width: parent.width - value.width
                            text: qsTr("Straighten")
                            color: Theme.ink2
                            font.pixelSize: Theme.body - 1
                        }
                        Text {
                            id: value
                            text: Math.round(tiltSlider.value * 10) / 10 + "°"
                            color: Theme.ink
                            font.pixelSize: Theme.body - 1
                            font.features: ({ "tnum": 1 })
                        }
                    }
                    ValueSlider {
                        id: tiltSlider
                        width: parent.width
                        from: -45
                        to: 45
                        stepSize: 0.5
                        Accessible.name: qsTr("Straighten angle")
                        // Detection reruns on release, not on every step of the drag.
                        onPressedChanged: if (!pressed) Studio.setTilt(value)
                    }
                    Binding {
                        target: tiltSlider
                        property: "value"
                        value: Studio.tilt
                        when: !tiltSlider.pressed
                    }
                    Text {
                        width: parent.width
                        text: qsTr("Finds the faces again when you let go.")
                        wrapMode: Text.Wrap
                        color: Theme.ink3
                        font.pixelSize: Theme.caption
                    }
                }
            }
        }
        Divider {}
        IconButton {
            iconPath: Icons.undo
            tip: qsTr("Undo (Ctrl+Z)")
            enabled: Studio.canUndo
            onClicked: Studio.undo()
        }
        IconButton {
            iconPath: Icons.redo
            tip: qsTr("Redo (Ctrl+Y)")
            enabled: Studio.canRedo
            onClicked: Studio.redo()
        }
    }
}
