import QtQuick
import QtQuick.Layouts

Rectangle {
    id: root

    signal addPhotos

    implicitHeight: 56
    color: Theme.bg1

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: Theme.rule
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: Theme.pad
        anchors.rightMargin: Theme.pad
        spacing: 12

        Rectangle {
            Layout.preferredWidth: 30
            Layout.preferredHeight: 30
            radius: 8
            gradient: Gradient {
                GradientStop { position: 0; color: Theme.peach }
                GradientStop { position: 1; color: Theme.accent }
            }
            Accessible.ignored: true

            Icon {
                anchors.centerIn: parent
                path: Icons.faceScan
                color: Theme.textOnAccent
                size: 20
            }
        }

        Text {
            text: qsTr("Face Crop Studio")
            color: Theme.ink
            font.pixelSize: Theme.title
            font.weight: Font.DemiBold
        }

        Rectangle {
            implicitWidth: previewLabel.implicitWidth + 14
            implicitHeight: 20
            radius: 10
            color: Theme.surface2

            Text {
                id: previewLabel
                anchors.centerIn: parent
                text: qsTr("Qt preview")
                color: Theme.ink3
                font.pixelSize: Theme.caption
            }
        }

        Item { Layout.fillWidth: true }

        StatusPill {}

        AppButton {
            variant: "primary"
            iconPath: Icons.plus
            text: qsTr("Add photos")
            onClicked: root.addPhotos()
        }
    }
}
