import QtQuick
import QtQuick.Effects

Rectangle {
    id: root

    property string url

    function show(message: string, folderUrl: string) {
        label.text = message
        url = folderUrl
        state = "shown"
        timer.restart()
    }

    implicitWidth: row.implicitWidth + 16
    implicitHeight: 48
    radius: 12
    color: Theme.surface2
    border.width: 1
    border.color: Theme.rule
    opacity: 0
    visible: opacity > 0
    transform: Translate { id: shift; y: 14 }
    layer.enabled: visible
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowColor: "#aa000000"
        shadowBlur: 0.9
        shadowVerticalOffset: 8
    }
    Accessible.role: Accessible.AlertMessage
    Accessible.name: label.text

    states: State {
        name: "shown"
        PropertyChanges { root.opacity: 1; shift.y: 0 }
    }
    transitions: [
        Transition {
            to: "shown"
            NumberAnimation { properties: "opacity,y"; duration: Theme.medium; easing.type: Easing.OutCubic }
        },
        Transition {
            from: "shown"
            NumberAnimation { properties: "opacity,y"; duration: Theme.fast + 40; easing.type: Easing.InCubic }
        }
    ]

    Timer {
        id: timer
        interval: 6000
        onTriggered: root.state = ""
    }
    HoverHandler {
        onHoveredChanged: hovered ? timer.stop() : timer.restart()
    }

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 10

        Icon {
            anchors.verticalCenter: parent.verticalCenter
            path: root.url !== "" ? Icons.check : Icons.info
            color: root.url !== "" ? Theme.green : Theme.ink2
        }
        Text {
            id: label
            anchors.verticalCenter: parent.verticalCenter
            rightPadding: root.url !== "" ? 4 : 8
            color: Theme.ink
            font.pixelSize: Theme.body
        }
        AppButton {
            anchors.verticalCenter: parent.verticalCenter
            visible: root.url !== ""
            implicitHeight: 32
            iconPath: Icons.folder
            text: qsTr("Open folder")
            onClicked: {
                Qt.openUrlExternally(root.url)
                root.state = ""
            }
        }
    }
}
