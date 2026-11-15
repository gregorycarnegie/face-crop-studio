pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

// Names exported faces from a spreadsheet: one column of photo file names, one of names.
Item {
    id: root

    readonly property var mapping: Studio.mapping
    readonly property bool loaded: mapping.loaded === true

    FileDialog {
        id: picker
        title: qsTr("Choose a mapping file")
        nameFilters: [qsTr("Mapping files (*.csv *.tsv *.xlsx *.xls *.xlsm *.ods *.parquet *.pq *.db *.sqlite *.sqlite3)"),
                      qsTr("All files (*)")]
        onAccepted: Studio.loadMapping(selectedFile.toString())
    }

    // Nothing loaded: what this is for, and how to start.
    ColumnLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: Theme.pad
        anchors.topMargin: 24
        visible: !root.loaded
        spacing: 12

        Rectangle {
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: 56
            Layout.preferredHeight: 56
            radius: 16
            color: Qt.alpha(Theme.accent, 0.10)
            border.width: 1
            border.color: Qt.alpha(Theme.accent, 0.35)
            Accessible.ignored: true

            Icon {
                anchors.centerIn: parent
                path: "M4 5h16v14H4zM4 10h16M10 10v9"
                color: Theme.accent
                size: 26
            }
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Name exports from a spreadsheet")
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.Wrap
            color: Theme.ink
            font.pixelSize: Theme.title
            font.weight: Font.DemiBold
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("A CSV, Excel, Parquet or SQLite table with a column of photo file names and a column of output names. Each matching photo's crops are saved under its name.")
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.Wrap
            lineHeight: 1.3
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
        AppButton {
            Layout.alignment: Qt.AlignHCenter
            Layout.topMargin: 4
            iconPath: Icons.folder
            text: qsTr("Choose mapping file")
            onClicked: picker.open()
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Or drop it on the window.")
            horizontalAlignment: Text.AlignHCenter
            color: Qt.alpha(Theme.ink3, 0.8)
            font.pixelSize: Theme.caption
        }
    }

    // A mapping is loaded: its columns, what it matched, and a look at its rows.
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Theme.pad
        anchors.topMargin: 6
        visible: root.loaded
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Icon {
                path: "M4 5h16v14H4zM4 10h16M10 10v9"
                color: Theme.ink2
            }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 0

                Text {
                    Layout.fillWidth: true
                    text: root.mapping.file ?? ""
                    elide: Text.ElideMiddle
                    color: Theme.ink
                    font.pixelSize: Theme.body
                    font.weight: Font.Medium
                }
                Text {
                    text: qsTr("%1 · %2 rows").arg(root.mapping.format ?? "").arg(root.mapping.rows ?? 0)
                    color: Theme.ink3
                    font.pixelSize: Theme.caption
                }
            }
            IconButton {
                side: 30
                iconPath: Icons.close
                tip: qsTr("Stop using this mapping")
                onClicked: Studio.clearMapping()
            }
        }

        AppComboBox {
            Layout.fillWidth: true
            visible: (root.mapping.sheets ?? []).length > 1
            model: root.mapping.sheets ?? []
            currentIndex: model.indexOf(root.mapping.sheet ?? "")
            Accessible.name: qsTr("Sheet")
            onActivated: index => Studio.setMappingSheet(model[index])
        }
        AppComboBox {
            Layout.fillWidth: true
            visible: (root.mapping.tables ?? []).length > 1
            model: root.mapping.tables ?? []
            currentIndex: model.indexOf(root.mapping.table ?? "")
            Accessible.name: qsTr("Table")
            onActivated: index => Studio.setMappingTable(model[index])
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 8
            rowSpacing: 4

            Text {
                text: qsTr("Photo column")
                color: Theme.ink2
                font.pixelSize: Theme.caption
            }
            Text {
                text: qsTr("Name column")
                color: Theme.ink2
                font.pixelSize: Theme.caption
            }
            AppComboBox {
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                model: root.mapping.columns ?? []
                currentIndex: root.mapping.source ?? 0
                Accessible.name: qsTr("Photo column")
                onActivated: index => Studio.setMappingColumns(index, root.mapping.output)
            }
            AppComboBox {
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                model: root.mapping.columns ?? []
                currentIndex: root.mapping.output ?? 0
                Accessible.name: qsTr("Name column")
                onActivated: index => Studio.setMappingColumns(root.mapping.source, index)
            }
        }

        Rectangle {
            readonly property bool all: root.mapping.photos > 0 && root.mapping.matched === root.mapping.photos
            Layout.fillWidth: true
            implicitHeight: matchLabel.implicitHeight + 16
            radius: Theme.radiusSmall
            color: all ? Qt.alpha(Theme.green, 0.12) : Theme.surface
            border.width: 1
            border.color: all ? Qt.alpha(Theme.green, 0.4) : Theme.rule

            Text {
                id: matchLabel
                anchors.fill: parent
                anchors.margins: 8
                text: root.mapping.photos === 0
                    ? qsTr("Add photos to match them against this file.")
                    : qsTr("%1 of %2 photos matched").arg(root.mapping.matched).arg(root.mapping.photos)
                wrapMode: Text.Wrap
                color: parent.all ? Theme.green : Theme.ink2
                font.pixelSize: Theme.caption
                font.weight: Font.Medium
            }
        }

        Text {
            text: qsTr("First rows")
            color: Theme.ink3
            font.pixelSize: Theme.caption
            font.weight: Font.DemiBold
            font.letterSpacing: 0.9
            font.capitalization: Font.AllUppercase
        }
        ListView {
            id: rows

            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 1
            model: root.mapping.preview ?? []
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}

            delegate: Rectangle {
                id: row

                required property var modelData
                required property int index

                width: ListView.view.width
                height: 28
                color: index % 2 ? "transparent" : Qt.alpha(Theme.surface, 0.6)
                radius: 4

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 6

                    Text {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        text: row.modelData[root.mapping.source] ?? ""
                        elide: Text.ElideMiddle
                        color: Theme.ink2
                        font.pixelSize: Theme.caption
                    }
                    Text {
                        text: "→"
                        color: Theme.ink3
                        font.pixelSize: Theme.caption
                    }
                    Text {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        text: row.modelData[root.mapping.output] ?? ""
                        elide: Text.ElideRight
                        color: Theme.ink
                        font.pixelSize: Theme.caption
                    }
                }
            }
        }
    }
}
