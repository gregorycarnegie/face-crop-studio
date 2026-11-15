import QtQuick
import QtQuick.Controls.Basic

Button {
    id: control

    property string iconPath
    // "primary", "secondary" or "ghost"
    property string variant: "secondary"
    // 0..1 fills the button as a progress bar; negative hides it.
    property real progress: -1
    readonly property bool primary: variant === "primary"
    readonly property color foreground: primary ? Theme.textOnAccent : Theme.ink

    implicitHeight: 36
    leftPadding: 14
    rightPadding: 14
    hoverEnabled: true
    opacity: enabled ? 1 : 0.4
    font.pixelSize: Theme.body
    font.weight: Font.DemiBold

    contentItem: Item {
        implicitWidth: row.implicitWidth
        implicitHeight: row.implicitHeight

        Row {
            id: row
            anchors.centerIn: parent
            spacing: 8

            Icon {
                anchors.verticalCenter: parent.verticalCenter
                visible: control.iconPath !== ""
                path: control.iconPath
                color: control.foreground
                size: 16
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: control.text
                font: control.font
                color: control.foreground
            }
        }
    }

    background: Rectangle {
        radius: Theme.radiusSmall
        color: control.primary ? (control.down ? Theme.accentDown : control.hovered ? Theme.accentHover : Theme.accent)
             : control.variant === "ghost" ? (control.down ? Theme.rule : control.hovered ? Theme.surface2 : "transparent")
             : (control.down ? Theme.rule : control.hovered ? Theme.surface2 : Theme.surface)
        border.width: control.variant === "secondary" ? 1 : 0
        border.color: Theme.rule

        Behavior on color { ColorAnimation { duration: Theme.fast } }

        Rectangle {
            visible: control.progress >= 0
            width: parent.width * Math.min(1, Math.max(0, control.progress))
            height: parent.height
            radius: parent.radius
            color: Qt.alpha(Theme.textOnAccent, 0.2)
        }

        Rectangle {
            anchors.fill: parent
            anchors.margins: -3
            radius: parent.radius + 3
            color: "transparent"
            border.width: 2
            border.color: Theme.focus
            visible: control.visualFocus
        }
    }
}
