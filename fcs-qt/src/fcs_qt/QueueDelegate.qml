import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

ItemDelegate {
    id: control

    required property int index
    required property string name
    required property string thumb
    required property string status
    required property string error
    required property int faces
    required property int selected
    readonly property bool current: ListView.isCurrentItem

    implicitHeight: 64
    padding: 8
    hoverEnabled: true
    Accessible.name: name
    ToolTip.visible: hovered && error !== ""
    ToolTip.text: error
    ToolTip.delay: 400

    background: Rectangle {
        radius: Theme.radius
        color: control.current ? Theme.surface2 : control.hovered ? Theme.surface : "transparent"
        border.width: control.visualFocus ? 2 : 0
        border.color: Theme.focus

        Behavior on color { ColorAnimation { duration: Theme.fast } }

        Rectangle {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            width: 3
            height: control.current ? parent.height - 28 : 0
            radius: 1.5
            color: Theme.accent
        }
    }

    contentItem: RowLayout {
        spacing: 12

        Rectangle {
            Layout.preferredWidth: 48
            Layout.preferredHeight: 48
            radius: 8
            color: Theme.surface2
            clip: true

            Image {
                id: thumbnail
                anchors.fill: parent
                source: control.thumb
                sourceSize: Qt.size(96, 96)
                fillMode: Image.PreserveAspectCrop
                asynchronous: true
                opacity: status === Image.Ready ? 1 : 0

                Behavior on opacity { OpacityAnimator { duration: Theme.medium } }
            }
            Spinner {
                anchors.centerIn: parent
                visible: control.status === "queued"
                size: 16
            }
            Icon {
                anchors.centerIn: parent
                visible: control.status === "failed"
                path: Icons.alert
                color: Theme.red
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2

            Text {
                Layout.fillWidth: true
                text: control.name
                elide: Text.ElideMiddle
                color: Theme.ink
                font.pixelSize: Theme.body
            }
            Text {
                Layout.fillWidth: true
                elide: Text.ElideRight
                text: control.status === "queued" ? qsTr("Finding faces…")
                    : control.status === "failed" ? qsTr("Couldn't read this file")
                    : control.faces === 0 ? qsTr("No faces found")
                    : control.faces === 1 ? qsTr("1 face") : qsTr("%1 faces").arg(control.faces)
                color: control.status === "failed" ? Theme.red : Theme.ink3
                font.pixelSize: Theme.caption
            }
        }

        Rectangle {
            visible: control.status === "done" && control.faces > 0
            implicitWidth: badge.implicitWidth + 14
            implicitHeight: 22
            radius: 11
            color: control.selected > 0 ? Qt.alpha(Theme.accent, 0.16) : Theme.surface2

            Text {
                id: badge
                anchors.centerIn: parent
                text: qsTr("%1/%2").arg(control.selected).arg(control.faces)
                color: control.selected > 0 ? Theme.peach : Theme.ink3
                font.pixelSize: Theme.caption
                font.weight: Font.DemiBold
                font.features: ({ "tnum": 1 })
            }
        }
    }
}
