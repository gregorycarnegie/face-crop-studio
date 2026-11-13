import QtQuick
import QtQuick.Layouts

ColumnLayout {
    id: root

    property string title
    property string detail
    default property alias content: body.data

    spacing: 10

    RowLayout {
        Layout.fillWidth: true

        Text {
            text: root.title
            color: Theme.ink3
            font.pixelSize: Theme.caption
            font.weight: Font.DemiBold
            font.letterSpacing: 0.9
            font.capitalization: Font.AllUppercase
            Accessible.role: Accessible.Heading
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root.detail
            color: Theme.ink2
            font.pixelSize: Theme.caption
            font.features: ({ "tnum": 1 })
        }
    }

    ColumnLayout {
        id: body
        Layout.fillWidth: true
        spacing: 8
    }
}
