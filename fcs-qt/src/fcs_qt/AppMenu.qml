pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic

// A themed pop-up menu; fill it with Action items.
Menu {
    id: menu

    padding: 4
    font.pixelSize: Theme.body

    delegate: MenuItem {
        id: item

        implicitHeight: 34
        leftPadding: 12
        rightPadding: 16
        hoverEnabled: true

        contentItem: Text {
            text: item.text
            font: item.font
            color: Theme.ink
            opacity: item.enabled ? 1 : 0.4
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }

        background: Rectangle {
            radius: Theme.radiusSmall - 2
            color: item.highlighted ? Theme.surface2 : "transparent"
        }
    }

    background: Rectangle {
        implicitWidth: 220
        radius: Theme.radiusSmall
        color: Theme.surface
        border.width: 1
        border.color: Theme.rule
    }
}
