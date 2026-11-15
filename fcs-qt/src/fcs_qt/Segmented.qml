pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

// One-of-n picker. `options` is [{ value, label, glyph?, enabled?, tip? }]; glyph draws a shape
// outline ("rectangle", "rounded" or "ellipse") before the label; `enabled: false` greys an
// option out, with `tip` saying why.
Rectangle {
    id: root

    property var options: []
    property string value
    signal activated(string value)

    implicitHeight: 40
    radius: Theme.radius
    color: Theme.surface
    border.width: 1
    border.color: Theme.rule
    Accessible.role: Accessible.Grouping

    RowLayout {
        anchors.fill: parent
        anchors.margins: 3
        spacing: 3

        Repeater {
            model: root.options

            delegate: AbstractButton {
                id: segment

                required property var modelData
                readonly property bool active: root.value === modelData.value

                Layout.fillWidth: true
                Layout.fillHeight: true
                hoverEnabled: true
                enabled: root.enabled && (modelData.enabled ?? true)
                opacity: enabled ? 1 : 0.4
                ToolTip.visible: hovered && (modelData.tip ?? "") !== ""
                ToolTip.delay: 400
                ToolTip.text: modelData.tip ?? ""
                Accessible.role: Accessible.RadioButton
                Accessible.name: modelData.label
                Accessible.checked: active
                onClicked: root.activated(modelData.value)

                background: Rectangle {
                    radius: Theme.radius - 3
                    color: segment.active ? Theme.surface2
                         : segment.hovered ? Qt.alpha(Theme.surface2, 0.5) : "transparent"
                    border.width: segment.visualFocus ? 2 : segment.active ? 1 : 0
                    border.color: segment.visualFocus ? Theme.focus : Theme.rule

                    Behavior on color { ColorAnimation { duration: Theme.fast } }
                }

                contentItem: Item {
                    implicitWidth: content.implicitWidth
                    implicitHeight: content.implicitHeight

                    Row {
                        id: content
                        anchors.centerIn: parent
                        spacing: 7

                        Rectangle {
                            readonly property string glyph: segment.modelData.glyph ?? ""
                            anchors.verticalCenter: parent.verticalCenter
                            visible: glyph !== ""
                            width: glyph === "ellipse" ? 14 : 17
                            height: 14
                            radius: glyph === "ellipse" ? 7 : glyph === "rounded" ? 4.5 : 1.5
                            color: "transparent"
                            border.width: 1.6
                            border.color: segment.active ? Theme.accent : Theme.ink3
                        }
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: segment.modelData.label
                            color: segment.active ? Theme.ink : Theme.ink3
                            font.pixelSize: Theme.body - 1
                            font.weight: segment.active ? Font.DemiBold : Font.Normal
                        }
                    }
                }
            }
        }
    }
}
