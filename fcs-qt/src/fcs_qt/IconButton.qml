import QtQuick
import QtQuick.Controls.Basic

// A square icon button with a tooltip. `active` marks a tool that is on.
AbstractButton {
    id: control

    property string iconPath
    property string tip
    property bool active
    property int side: 36

    implicitWidth: side
    implicitHeight: side
    hoverEnabled: true
    focusPolicy: Qt.TabFocus
    opacity: enabled ? 1 : 0.35
    Accessible.role: Accessible.Button
    Accessible.name: tip
    ToolTip.visible: hovered && tip !== ""
    ToolTip.delay: 400
    ToolTip.text: tip

    background: Rectangle {
        radius: Theme.radiusSmall
        color: control.active ? Qt.alpha(Theme.accent, 0.18)
             : control.down ? Theme.rule : control.hovered ? Theme.surface2 : "transparent"
        border.width: control.visualFocus ? 2 : control.active ? 1 : 0
        border.color: control.visualFocus ? Theme.focus : Theme.accent

        Behavior on color { ColorAnimation { duration: Theme.fast } }
    }

    contentItem: Item {
        Icon {
            anchors.centerIn: parent
            path: control.iconPath
            color: control.active ? Theme.accent : Theme.ink2
            size: 18
        }
    }
}
