pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts

// The camera, over the canvas: a live picture, and a shutter that adds the frame to the photos.
Rectangle {
    id: root

    readonly property bool streaming: Studio.cameraUrl !== ""

    color: Theme.bg

    Shortcut {
        sequence: "Space"
        enabled: root.visible && root.streaming
        onActivated: Studio.snapshot()
    }
    Shortcut {
        sequence: "Escape"
        enabled: root.visible
        onActivated: Studio.closeCamera()
    }

    // Starting, or why there is no picture.
    Column {
        anchors.centerIn: parent
        visible: !root.streaming
        spacing: 14

        Spinner {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: Studio.cameraStatus.startsWith(qsTr("Starting"))
            size: 28
        }
        Icon {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: !Studio.cameraStatus.startsWith(qsTr("Starting"))
            path: Icons.camera
            color: Theme.ink3
            size: 40
        }
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.min(implicitWidth, root.width - 80)
            text: Studio.cameraStatus
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.Wrap
            color: Theme.ink2
            font.pixelSize: Theme.title
        }
    }

    Image {
        id: feed

        anchors.fill: parent
        anchors.margins: 32
        anchors.bottomMargin: 112
        visible: root.streaming
        source: Studio.cameraUrl
        fillMode: Image.PreserveAspectFit
        // Each frame replaces the last at once: an asynchronous load would blank between them.
        asynchronous: false
        cache: false
        smooth: true
        Accessible.role: Accessible.Graphic
        Accessible.name: qsTr("Camera picture")

        Item {
            x: (feed.width - feed.paintedWidth) / 2
            y: (feed.height - feed.paintedHeight) / 2
            width: feed.paintedWidth
            height: feed.paintedHeight

            Repeater {
                model: Studio.cameraFaces
                delegate: Rectangle {
                    required property var modelData
                    x: modelData.x * parent.width
                    y: modelData.y * parent.height
                    width: modelData.w * parent.width
                    height: modelData.h * parent.height
                    radius: 6
                    color: Qt.alpha(Theme.accent, 0.08)
                    border.width: 2
                    border.color: Theme.accent
                }
            }
        }
    }

    // The controls.
    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        implicitWidth: bar.implicitWidth + 28
        implicitHeight: 80
        radius: 40
        color: Theme.surface
        border.width: 1
        border.color: Theme.rule
        layer.enabled: true
        layer.effect: MultiEffect {
            shadowEnabled: true
            shadowColor: "#99000000"
            shadowBlur: 0.8
            shadowVerticalOffset: 6
        }

        RowLayout {
            id: bar
            anchors.centerIn: parent
            spacing: 16

            AppComboBox {
                Layout.preferredWidth: 200
                visible: Studio.cameras.length > 1
                model: Studio.cameras.map(c => c.name)
                Accessible.name: qsTr("Camera")
                onActivated: index => Studio.openCamera(index)
            }
            ToggleRow {
                Layout.preferredWidth: 190
                text: qsTr("Find faces live")
                hint: qsTr("Detect faces in every frame; the snapshot is detected either way")
                enabled: root.streaming && Studio.ready
                on: Studio.cameraLive
                onToggledTo: on => Studio.setCameraLive(on)
            }

            // The shutter.
            AbstractButton {
                id: shutter

                Layout.preferredWidth: 60
                Layout.preferredHeight: 60
                enabled: root.streaming
                hoverEnabled: true
                opacity: enabled ? 1 : 0.4
                Accessible.role: Accessible.Button
                Accessible.name: qsTr("Take snapshot")
                ToolTip.visible: hovered
                ToolTip.delay: 400
                ToolTip.text: qsTr("Take snapshot (Space); it joins the photos")
                onClicked: Studio.snapshot()

                background: Rectangle {
                    radius: width / 2
                    color: "transparent"
                    border.width: 3
                    border.color: shutter.visualFocus ? Theme.focus : Theme.ink

                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 6
                        radius: width / 2
                        color: shutter.down ? Theme.accentDown : shutter.hovered ? Theme.accentHover : Theme.accent
                        scale: shutter.down ? 0.92 : 1

                        Behavior on scale { ScaleAnimator { duration: Theme.fast } }
                    }
                }
            }

            AppButton {
                variant: "secondary"
                text: qsTr("Done")
                onClicked: Studio.closeCamera()
            }
        }
    }
}
