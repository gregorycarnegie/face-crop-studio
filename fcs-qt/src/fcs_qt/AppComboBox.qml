pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic

// A themed drop-down over a list of strings.
ComboBox {
    id: control

    implicitHeight: 36
    leftPadding: 12
    rightPadding: 36
    hoverEnabled: true
    font.pixelSize: Theme.body

    contentItem: Text {
        text: control.displayText
        font: control.font
        color: Theme.ink
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Icon {
        x: control.width - width - 12
        y: (control.height - height) / 2
        path: "M7 10l5 5 5-5"
        color: Theme.ink3
        size: 16
    }

    background: Rectangle {
        radius: Theme.radiusSmall
        color: control.hovered ? Theme.surface2 : Theme.surface
        border.width: control.visualFocus || control.popup.visible ? 2 : 1
        border.color: control.visualFocus || control.popup.visible ? Theme.accent : Theme.rule

        Behavior on color { ColorAnimation { duration: Theme.fast } }
    }

    delegate: ItemDelegate {
        id: option

        required property int index
        required property var modelData

        width: control.width - 8
        x: 4
        implicitHeight: 32
        highlighted: control.highlightedIndex === index
        hoverEnabled: true

        contentItem: Text {
            text: option.modelData
            color: option.index === control.currentIndex ? Theme.peach : Theme.ink
            font.pixelSize: Theme.body - 1
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: Theme.radiusSmall - 2
            color: option.highlighted ? Theme.surface2 : "transparent"
        }
    }

    popup: Popup {
        y: control.height + 4
        width: control.width
        implicitHeight: Math.min(contentItem.implicitHeight + 8, 280)
        padding: 4

        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle {
            radius: Theme.radiusSmall + 2
            color: Theme.surface
            border.width: 1
            border.color: Theme.rule
        }
    }
}
