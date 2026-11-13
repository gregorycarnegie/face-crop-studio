import QtQuick
import QtQuick.Layouts

// A titled slider bound to a setting. It follows `value` from outside except while dragged,
// and reports user moves through `moved(value)`.
ColumnLayout {
    id: root

    property string title
    property real value
    property real from: 0
    property real to: 1
    property real stepSize: 0
    // Turns the slider value into the text shown beside the title.
    property var format: v => v.toFixed(2)
    // Report only when the drag ends, for settings too costly to apply on every step.
    property bool commitOnRelease: false
    signal moved(real value)

    spacing: 2

    RowLayout {
        Layout.fillWidth: true

        Text {
            Layout.fillWidth: true
            text: root.title
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
            elide: Text.ElideRight
        }
        Text {
            text: root.format(slider.value)
            color: Theme.ink
            font.pixelSize: Theme.body - 1
            font.features: ({ "tnum": 1 })
        }
    }

    ValueSlider {
        id: slider
        Layout.fillWidth: true
        from: root.from
        to: root.to
        stepSize: root.stepSize
        Accessible.name: root.title
        // Keyboard steps arrive unpressed, so they still commit at once.
        onMoved: if (!root.commitOnRelease || !pressed) root.moved(value)
        onPressedChanged: if (root.commitOnRelease && !pressed) root.moved(value)
    }

    Binding {
        target: slider
        property: "value"
        value: root.value
        when: !slider.pressed
        // Not the default restore: a key press also sets `pressed`, and restoring would put
        // back the slider's value from before this binding (its minimum) mid-step.
        restoreMode: Binding.RestoreNone
    }
}
