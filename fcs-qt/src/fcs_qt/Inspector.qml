pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    signal exportRequested

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

        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: Theme.pad
            spacing: 22

            Section {
                Layout.fillWidth: true
                title: qsTr("Output size")
                detail: qsTr("%1 × %2 px").arg(Studio.outputWidth).arg(Studio.outputHeight)

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 8
                    columnSpacing: 8

                    Repeater {
                        model: Studio.presets
                        delegate: PresetCard {
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                            onClicked: Studio.setPreset(modelData.name)
                        }
                    }
                }
            }

            Section {
                Layout.fillWidth: true
                title: qsTr("Face size")
                detail: qsTr("%1% of height").arg(Math.round(faceSlider.value))

                ValueSlider {
                    id: faceSlider
                    Layout.fillWidth: true
                    from: 30
                    to: 90
                    stepSize: 1
                    Accessible.name: qsTr("Face size")
                    onMoved: Studio.setFaceHeight(value)
                }
                // Follows presets without fighting the user's drag.
                Binding {
                    target: faceSlider
                    property: "value"
                    value: Studio.faceHeight
                    when: !faceSlider.pressed
                }
            }

            Section {
                Layout.fillWidth: true
                title: qsTr("Shape")

                Segmented {
                    Layout.fillWidth: true
                    value: Studio.shape
                    options: [
                        { value: "rectangle", label: qsTr("Square"), glyph: "rectangle" },
                        { value: "rounded", label: qsTr("Rounded"), glyph: "rounded" },
                        { value: "ellipse", label: qsTr("Oval"), glyph: "ellipse" }
                    ]
                    onActivated: value => Studio.setShape(value)
                }
            }

            Section {
                Layout.fillWidth: true
                title: qsTr("Format")

                Segmented {
                    Layout.fillWidth: true
                    value: Studio.format
                    options: [
                        { value: "png", label: "PNG" },
                        { value: "jpg", label: "JPEG" },
                        { value: "webp", label: "WebP" }
                    ]
                    onActivated: value => Studio.setFormat(value)
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: Theme.rule
        }

        Section {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: Theme.pad
            Layout.bottomMargin: 0
            title: qsTr("Preview")
            detail: Studio.previews.length > 0 ? qsTr("as exported") : ""

            GridView {
                id: grid

                readonly property real aspect: Studio.outputHeight / Math.max(1, Studio.outputWidth)

                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                cellWidth: width / 2
                cellHeight: Math.min(cellWidth * aspect, 280) + 30
                model: Studio.previews
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar {}

                delegate: PreviewCard {
                    width: grid.cellWidth - 8
                    height: grid.cellHeight - 8
                }

                Text {
                    anchors.centerIn: parent
                    width: parent.width - 32
                    visible: grid.count === 0
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
            Layout.topMargin: 12
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
