import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: root

    signal addPhotos

    spacing: 14

    Rectangle {
        Layout.alignment: Qt.AlignHCenter
        Layout.preferredWidth: 92
        Layout.preferredHeight: 92
        Layout.bottomMargin: 6
        radius: 26
        color: Qt.alpha(Theme.accent, 0.10)
        border.width: 1
        border.color: Qt.alpha(Theme.accent, 0.35)
        Accessible.ignored: true

        Icon {
            anchors.centerIn: parent
            path: Icons.faceScan
            color: Theme.accent
            size: 46
        }
    }

    Text {
        Layout.alignment: Qt.AlignHCenter
        text: qsTr("Drop photos to find faces")
        color: Theme.ink
        font.pixelSize: Theme.display
        font.weight: Font.DemiBold
    }

    Text {
        Layout.alignment: Qt.AlignHCenter
        Layout.maximumWidth: 400
        text: qsTr("Add photos or a whole folder. Detection runs on this computer; nothing is uploaded.")
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.Wrap
        lineHeight: 1.35
        color: Theme.ink3
        font.pixelSize: Theme.body
    }

    AppButton {
        Layout.alignment: Qt.AlignHCenter
        Layout.topMargin: 8
        implicitHeight: 40
        variant: "primary"
        iconPath: Icons.plus
        text: qsTr("Choose photos")
        onClicked: root.addPhotos()
    }

    Text {
        Layout.alignment: Qt.AlignHCenter
        text: qsTr("JPEG · PNG · WebP · TIFF · HEIC · camera RAW")
        color: Qt.alpha(Theme.ink3, 0.8)
        font.pixelSize: Theme.caption
    }
}
