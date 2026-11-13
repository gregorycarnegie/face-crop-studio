import QtQuick
import QtQuick.Controls.Basic

// A labelled switch showing the bound setting `on`. It never flips itself: a click reports
// `toggledTo(!on)` and the setting coming back moves it, so a rejected or reset value shows.
Switch {
    id: control

    property bool on
    property string hint
    signal toggledTo(bool on)

    checkable: false
    checked: on
    // Not checkable (see above), so say what it is ourselves: a screen reader would otherwise
    // announce a plain button with no state.
    Accessible.role: Accessible.CheckBox
    Accessible.checkable: true
    Accessible.checked: on
    Accessible.name: text
    implicitHeight: 34
    padding: 0
    hoverEnabled: true
    font.pixelSize: Theme.body
    onClicked: toggledTo(!on)

    ToolTip.visible: hovered && hint !== ""
    ToolTip.delay: 500
    ToolTip.text: hint

    contentItem: Text {
        rightPadding: control.indicator.width + 10
        text: control.text
        font: control.font
        color: Theme.ink
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Rectangle {
        x: control.width - width
        anchors.verticalCenter: parent.verticalCenter
        implicitWidth: 38
        implicitHeight: 22
        radius: 11
        color: control.checked ? Theme.accent : Theme.surface2
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? Theme.focus : control.checked ? Theme.accent : Theme.rule

        Behavior on color { ColorAnimation { duration: Theme.fast } }

        Rectangle {
            x: control.checked ? parent.width - width - 3 : 3
            anchors.verticalCenter: parent.verticalCenter
            width: 16
            height: 16
            radius: 8
            color: control.checked ? Theme.textOnAccent : Theme.ink2

            Behavior on x { NumberAnimation { duration: Theme.fast; easing.type: Easing.OutCubic } }
        }
    }
}
