import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

// A titled whole-number field. Commits on Enter or when focus leaves; follows `value`
// otherwise.
ColumnLayout {
    id: root

    property string title
    property int value
    property int from: 1
    property int to: 8192
    property string suffix
    signal committed(int value)

    spacing: 4

    Text {
        text: root.title
        color: Theme.ink2
        font.pixelSize: Theme.body - 1
    }

    TextField {
        id: field

        function commit() {
            const v = parseInt(text)
            if (!isNaN(v) && v !== root.value)
                root.committed(Math.max(root.from, Math.min(root.to, v)))
            else
                text = root.value
        }

        Layout.fillWidth: true
        implicitHeight: 36
        leftPadding: 12
        rightPadding: suffixLabel.implicitWidth + 16
        text: root.value
        validator: IntValidator { bottom: root.from; top: root.to }
        inputMethodHints: Qt.ImhDigitsOnly
        selectByMouse: true
        color: Theme.ink
        selectionColor: Qt.alpha(Theme.accent, 0.4)
        selectedTextColor: Theme.ink
        font.pixelSize: Theme.body
        font.features: ({ "tnum": 1 })
        Accessible.name: root.title
        onEditingFinished: commit()
        onActiveFocusChanged: if (!activeFocus) commit()

        background: Rectangle {
            radius: Theme.radiusSmall
            color: Theme.surface
            border.width: field.activeFocus ? 2 : 1
            border.color: field.activeFocus ? Theme.accent : field.hovered ? Theme.rule2 : Theme.rule
        }

        Text {
            id: suffixLabel
            anchors.right: parent.right
            anchors.rightMargin: 12
            anchors.verticalCenter: parent.verticalCenter
            text: root.suffix
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
    }

    // A new value from outside (a preset) replaces what is typed, unless it is being typed.
    Binding {
        target: field
        property: "text"
        value: root.value
        when: !field.activeFocus
    }
}
