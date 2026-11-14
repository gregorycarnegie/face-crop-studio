import QtQuick
import QtQuick.Controls
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
            iconPath: Icons.camera
            text: Studio.cameraOpen ? qsTr("Close camera") : qsTr("Camera")
            onClicked: Studio.cameraOpen ? Studio.closeCamera() : Studio.openCamera(0)
        }

        AppButton {
            variant: "primary"
            iconPath: Icons.plus
            text: qsTr("Add photos")
            onClicked: root.addPhotos()
        }

        IconButton {
            id: appMenuButton
            iconPath: Icons.more
            tip: qsTr("Settings and About")
            onClicked: appMenu.popup(appMenuButton, appMenuButton.width - appMenu.width, appMenuButton.height + 6)
        }
    }

    AppMenu {
        id: appMenu
        Action {
            text: qsTr("Restore default settings…")
            onTriggered: restoreDialog.open()
        }
        MenuSeparator {
            contentItem: Rectangle { implicitHeight: 1; color: Theme.rule }
        }
        Action {
            text: qsTr("About Face Crop Studio")
            onTriggered: aboutDialog.open()
        }
    }

    ChoiceDialog {
        id: restoreDialog
        title: qsTr("Restore default settings?")
        message: qsTr("Every setting goes back to how Face Crop Studio ships: size, framing, shape, enhancements, output and backend. Your photos stay; if the backend or confidence floor changes, their faces are found again and hand edits are lost.")
        choices: [
            { value: "restore", label: qsTr("Restore defaults"), variant: "primary" },
            { value: "cancel", label: qsTr("Cancel"), variant: "ghost" }
        ]
        onChosen: value => { if (value === "restore") Studio.restoreDefaults() }
    }

    ChoiceDialog {
        id: aboutDialog
        title: qsTr("Face Crop Studio %1").arg(Studio.version)
        message: qsTr("Face Crop Studio uses a neural network to detect faces and automatically crop portraits at scale. Detection runs on this computer; nothing is uploaded.")
            + "\n\n" + qsTr("This is the Qt preview of the app. It uses Qt %1 under the GNU LGPL v3; the licence notices ship beside the program.").arg(Studio.qtVersion)
        choices: [
            { value: "close", label: qsTr("Close"), variant: "primary" },
            { value: "website", label: qsTr("facecropstudio.com"), variant: "ghost" }
        ]
        onChosen: value => { if (value === "website") Qt.openUrlExternally("https://facecropstudio.com/") }
    }
}
