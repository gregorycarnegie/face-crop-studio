import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// One inspector tab: a scrolling column of sections.
ScrollView {
    id: root

    default property alias content: column.data

    contentWidth: availableWidth
    padding: Theme.pad
    rightPadding: Theme.pad + 4
    clip: true

    ColumnLayout {
        id: column
        width: root.availableWidth
        spacing: 24
    }
}
