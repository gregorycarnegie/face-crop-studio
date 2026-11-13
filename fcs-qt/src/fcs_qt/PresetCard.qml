import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

AbstractButton {
    id: control

    // { name, width, height } from Studio.presets
    required property var modelData
    readonly property bool active: Studio.settings.crop.preset === modelData.name

    implicitHeight: 58
    padding: 10
    hoverEnabled: true
    Accessible.role: Accessible.RadioButton
    Accessible.name: qsTr("%1, %2 by %3 pixels").arg(modelData.name).arg(modelData.width).arg(modelData.height)
    Accessible.checked: active

    contentItem: RowLayout {
        spacing: 10

        Item {
            Layout.preferredWidth: 26
            Layout.preferredHeight: 26

            // The preset's aspect ratio, drawn.
            Rectangle {
                readonly property real unit: 24 / Math.max(control.modelData.width, control.modelData.height)
                anchors.centerIn: parent
                width: control.modelData.width * unit
                height: control.modelData.height * unit
                radius: 3
                color: control.active ? Qt.alpha(Theme.accent, 0.22) : "transparent"
                border.width: 1.5
                border.color: control.active ? Theme.accent : Theme.rule2
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 1

            Text {
                Layout.fillWidth: true
                text: control.modelData.name
                elide: Text.ElideRight
                color: Theme.ink
                font.pixelSize: Theme.body
                font.weight: Font.Medium
            }
            Text {
                text: qsTr("%1 × %2").arg(control.modelData.width).arg(control.modelData.height)
                color: control.active ? Theme.peach : Theme.ink3
                font.pixelSize: Theme.caption
                font.features: ({ "tnum": 1 })
            }
        }
    }

    background: Rectangle {
        radius: Theme.radius
        color: control.active ? Qt.alpha(Theme.accent, 0.09) : control.hovered ? Theme.surface2 : Theme.surface
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? Theme.focus : control.active ? Theme.accent : Theme.rule

        Behavior on color { ColorAnimation { duration: Theme.fast } }
        Behavior on border.color { ColorAnimation { duration: Theme.fast } }
    }
}
