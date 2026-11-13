import QtQuick
import QtQuick.Effects

// Floating summary under the photo: how many faces are going out, and bulk select.
Rectangle {
    id: root

    readonly property int total: Studio.faces.length
    readonly property int chosen: Studio.faces.filter(face => face.selected).length

    implicitWidth: row.implicitWidth + 12
    implicitHeight: 44
    radius: 22
    color: Theme.surface
    border.width: 1
    border.color: Theme.rule
    layer.enabled: true
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowColor: "#99000000"
        shadowBlur: 0.8
        shadowVerticalOffset: 6
    }

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 4

        Text {
            anchors.verticalCenter: parent.verticalCenter
            leftPadding: 12
            rightPadding: 8
            text: root.total === 0 ? qsTr("No faces in this photo")
                : qsTr("%1 of %2 selected in this photo").arg(root.chosen).arg(root.total)
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
        }
        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            visible: root.total > 0
            width: 1
            height: 20
            color: Theme.rule
        }
        AppButton {
            anchors.verticalCenter: parent.verticalCenter
            visible: root.total > 0
            implicitHeight: 32
            variant: "ghost"
            text: qsTr("All")
            enabled: root.chosen < root.total
            onClicked: Studio.selectAllFaces(true)
        }
        AppButton {
            anchors.verticalCenter: parent.verticalCenter
            visible: root.total > 0
            implicitHeight: 32
            variant: "ghost"
            text: qsTr("None")
            enabled: root.chosen > 0
            onClicked: Studio.selectAllFaces(false)
        }
    }
}
