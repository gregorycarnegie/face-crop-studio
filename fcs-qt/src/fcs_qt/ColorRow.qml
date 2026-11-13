import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import QtQuick.Layouts

// A titled colour swatch for an { red, green, blue, alpha } setting; opens the system picker.
AbstractButton {
    id: control

    property string title
    // { red, green, blue, alpha }, 0-255 each
    property var rgba: ({ red: 0, green: 0, blue: 0, alpha: 255 })
    readonly property color color: Qt.rgba(rgba.red / 255, rgba.green / 255, rgba.blue / 255, rgba.alpha / 255)
    signal picked(var rgba)

    implicitHeight: 36
    hoverEnabled: true
    Accessible.role: Accessible.Button
    Accessible.name: qsTr("%1, choose colour").arg(title)
    onClicked: dialog.open()

    contentItem: RowLayout {
        spacing: 10

        Text {
            Layout.fillWidth: true
            text: control.title
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
            elide: Text.ElideRight
        }
        Text {
            text: control.rgba.alpha === 0 ? qsTr("Transparent")
                : control.color.toString().toUpperCase().slice(0, 7)
            color: Theme.ink3
            font.pixelSize: Theme.caption
            font.features: ({ "tnum": 1 })
        }
        // Checkerboard under the swatch shows transparency.
        Rectangle {
            Layout.preferredWidth: 44
            Layout.preferredHeight: 26
            radius: 6
            color: "#ffffff"
            border.width: control.visualFocus ? 2 : 1
            border.color: control.visualFocus ? Theme.focus : control.hovered ? Theme.rule2 : Theme.rule
            clip: true

            Grid {
                anchors.fill: parent
                anchors.margins: 1
                columns: 6
                Repeater {
                    model: 18
                    Rectangle {
                        required property int index
                        width: 7
                        height: 8
                        color: (index + Math.floor(index / 6)) % 2 ? "#c8c8c8" : "#ffffff"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                anchors.margins: 1
                radius: 5
                color: control.color
            }
        }
    }

    background: Item {}

    ColorDialog {
        id: dialog
        title: control.title
        selectedColor: control.color
        options: ColorDialog.ShowAlphaChannel
        onAccepted: control.picked({
            red: Math.round(selectedColor.r * 255),
            green: Math.round(selectedColor.g * 255),
            blue: Math.round(selectedColor.b * 255),
            alpha: Math.round(selectedColor.a * 255)
        })
    }
}
