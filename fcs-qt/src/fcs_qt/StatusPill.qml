import QtQuick
import QtQuick.Controls

Rectangle {
    id: root

    readonly property bool failed: Studio.engineError !== ""

    implicitWidth: row.implicitWidth + 22
    implicitHeight: 28
    radius: 14
    color: Theme.surface
    border.width: 1
    border.color: failed ? Qt.alpha(Theme.red, 0.6) : Theme.rule
    Accessible.role: Accessible.StaticText
    Accessible.name: label.text

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 8

        Spinner {
            anchors.verticalCenter: parent.verticalCenter
            visible: !Studio.ready && !root.failed
            size: 12
        }
        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            visible: Studio.ready || root.failed
            width: 8
            height: 8
            radius: 4
            color: root.failed ? Theme.red : Theme.green
        }
        Text {
            id: label
            anchors.verticalCenter: parent.verticalCenter
            text: root.failed ? qsTr("Models failed to load")
                : Studio.ready ? Studio.engineLabel : qsTr("Loading models…")
            color: Theme.ink2
            font.pixelSize: Theme.caption
        }
    }

    HoverHandler { id: hover }
    ToolTip.visible: hover.hovered && root.failed
    ToolTip.text: Studio.engineError
}
