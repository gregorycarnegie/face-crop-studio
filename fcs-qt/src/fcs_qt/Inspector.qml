pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    signal exportRequested

    readonly property var crop: Studio.settings.crop

    color: Theme.bg1

    Rectangle {
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        width: 1
        color: Theme.rule
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 1
        spacing: 0

        Segmented {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            Layout.bottomMargin: 4
            value: UiState.inspectorTab
            options: [
                { value: "crop", label: qsTr("Crop") },
                { value: "shape", label: qsTr("Shape") },
                { value: "enhance", label: qsTr("Enhance") },
                { value: "output", label: qsTr("Output") }
            ]
            onActivated: value => UiState.inspectorTab = value
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: ["crop", "shape", "enhance", "output"].indexOf(UiState.inspectorTab)

            CropTab {}
            ShapeTab {}
            EnhanceTab {}
            OutputTab {}
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: Theme.rule
        }

        Section {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            Layout.bottomMargin: 0
            title: qsTr("Preview")
            detail: qsTr("%1 × %2 px").arg(root.crop.output_width).arg(root.crop.output_height)

            ListView {
                id: strip

                readonly property real aspect: root.crop.output_height / Math.max(1, root.crop.output_width)

                Layout.fillWidth: true
                Layout.preferredHeight: 196
                orientation: ListView.Horizontal
                spacing: 8
                clip: true
                model: Studio.previews
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.horizontal: ScrollBar {}

                delegate: PreviewCard {
                    height: strip.height - 12
                    width: Math.min(strip.width, (height - 36) / strip.aspect + 16)
                }

                Text {
                    anchors.centerIn: parent
                    width: parent.width - 32
                    visible: strip.count === 0
                    text: Studio.faces.length === 0 ? qsTr("Previews of the selected faces appear here.")
                                                    : qsTr("Select a face on the photo to preview it.")
                    horizontalAlignment: Text.AlignHCenter
                    wrapMode: Text.Wrap
                    color: Theme.ink3
                    font.pixelSize: Theme.caption
                }
            }
        }

        Text {
            readonly property int photos: Studio.queue.filter(entry => entry.selected > 0).length
            Layout.fillWidth: true
            Layout.leftMargin: Theme.pad
            Layout.rightMargin: Theme.pad
            Layout.topMargin: 4
            visible: Studio.selectedTotal > 0
            text: photos === 1 ? qsTr("Selected faces from 1 photo") : qsTr("Selected faces from %1 photos").arg(photos)
            horizontalAlignment: Text.AlignHCenter
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }

        AppButton {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            Layout.topMargin: 8
            implicitHeight: 44
            variant: "primary"
            iconPath: Studio.exporting ? "" : Icons.upload
            text: Studio.exporting ? qsTr("Exporting… %1%").arg(Math.round(Studio.exportProgress * 100))
                : Studio.selectedTotal === 1 ? qsTr("Export 1 face")
                : qsTr("Export %1 faces").arg(Studio.selectedTotal)
            progress: Studio.exporting ? Studio.exportProgress : -1
            // Stays enabled while exporting so the progress fill is readable; clicks are ignored.
            enabled: Studio.exporting || (Studio.ready && Studio.selectedTotal > 0)
            onClicked: if (!Studio.exporting) root.exportRequested()
        }
    }
}
