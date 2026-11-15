pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

// A modal question with a few answers. `choices` is [{ value, label, variant }], the first
// being the default (Enter); Escape or a click outside answers nothing.
Dialog {
    id: root

    property string message
    property var choices: []
    signal chosen(string value)

    anchors.centerIn: Overlay.overlay
    // From the window, not the parent: a dialog declared in the sidebar would be sidebar-wide.
    width: Math.min(460, (Overlay.overlay ? Overlay.overlay.width : 524) - 64)
    modal: true
    focus: true
    padding: 22
    topPadding: 20
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    Overlay.modal: Rectangle { color: Qt.alpha(Theme.bg, 0.6) }

    enter: Transition {
        NumberAnimation { property: "opacity"; from: 0; to: 1; duration: Theme.fast + 40; easing.type: Easing.OutCubic }
        NumberAnimation { property: "scale"; from: 0.96; to: 1; duration: Theme.medium; easing.type: Easing.OutCubic }
    }
    exit: Transition {
        NumberAnimation { property: "opacity"; to: 0; duration: Theme.fast; easing.type: Easing.InCubic }
    }

    background: Rectangle {
        radius: Theme.radius + 4
        color: Theme.surface
        border.width: 1
        border.color: Theme.rule
    }

    // The title is drawn below, with the message; the style's own header bar would repeat it.
    header: null

    // Title and message in one column of explicit width: wrapped text sized from the dialog's
    // own implicit size loops.
    contentItem: Column {
        width: root.availableWidth
        spacing: 10

        Text {
            width: parent.width
            text: root.title
            color: Theme.ink
            font.pixelSize: Theme.title
            font.weight: Font.DemiBold
            wrapMode: Text.Wrap
            Accessible.role: Accessible.Heading
        }
        Text {
            width: parent.width
            text: root.message
            color: Theme.ink2
            font.pixelSize: Theme.body
            lineHeight: 1.3
            wrapMode: Text.Wrap
        }
    }

    footer: RowLayout {
        spacing: 8

        Item { Layout.fillWidth: true }
        Repeater {
            model: root.choices
            delegate: AppButton {
                required property var modelData
                required property int index
                Layout.bottomMargin: root.padding
                Layout.rightMargin: index === root.choices.length - 1 ? root.padding : 0
                variant: modelData.variant ?? "secondary"
                text: modelData.label
                focus: index === 0
                onClicked: {
                    root.close()
                    root.chosen(modelData.value)
                }
                Keys.onReturnPressed: clicked()
            }
        }
    }
}
