pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

InspectorPage {
    id: root

    readonly property var crop: Studio.settings.crop
    function set(fields) { Studio.update({ crop: fields }) }

    Section {
        Layout.fillWidth: true
        title: qsTr("Output size")
        detail: Studio.presetName === "" ? qsTr("Custom") : ""

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

        Text {
            Layout.topMargin: 6
            text: qsTr("Aspect ratio")
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
        }
        // fcs-gui's picks and sizes. "Free" is lit when the size matches none of them, and
        // picking it keeps the current size.
        Segmented {
            readonly property var sizes: ({ "1:1": [1024, 1024], "4:5": [1024, 1280], "3:4": [768, 1024] })

            Layout.fillWidth: true
            value: {
                const ratio = root.crop.output_width / root.crop.output_height
                for (const key in sizes) {
                    if (Math.abs(ratio - sizes[key][0] / sizes[key][1]) < 0.005)
                        return key
                }
                return "free"
            }
            options: [
                { value: "free", label: qsTr("Free") },
                { value: "1:1", label: "1:1" },
                { value: "4:5", label: "4:5" },
                { value: "3:4", label: "3:4" }
            ]
            onActivated: value => {
                if (value in sizes)
                    root.set({ output_width: sizes[value][0], output_height: sizes[value][1], preset: "Custom" })
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.topMargin: 4
            spacing: 8

            NumberField {
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                title: qsTr("Width")
                suffix: "px"
                value: root.crop.output_width
                onCommitted: v => root.set({ output_width: v, preset: "Custom" })
            }
            NumberField {
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                title: qsTr("Height")
                suffix: "px"
                value: root.crop.output_height
                onCommitted: v => root.set({ output_height: v, preset: "Custom" })
            }
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Framing")

        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Face size")
            from: 10
            to: 100
            stepSize: 1
            value: root.crop.face_height_pct
            format: v => qsTr("%1% of height").arg(Math.round(v))
            onMoved: v => root.set({ face_height_pct: v })
        }

        Text {
            Layout.topMargin: 6
            text: qsTr("Position")
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
        }
        Segmented {
            Layout.fillWidth: true
            value: root.crop.positioning_mode
            options: [
                { value: "center", label: qsTr("Center") },
                { value: "rule-of-thirds", label: qsTr("Thirds") },
                { value: "custom", label: qsTr("Custom") }
            ]
            onActivated: value => root.set({ positioning_mode: value })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.crop.positioning_mode === "custom"
            title: qsTr("Offset X")
            from: -1
            to: 1
            value: root.crop.horizontal_offset
            format: v => (v > 0 ? "+" : "") + v.toFixed(2)
            onMoved: v => root.set({ horizontal_offset: v })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.crop.positioning_mode === "custom"
            title: qsTr("Offset Y")
            from: -1
            to: 1
            value: root.crop.vertical_offset
            format: v => (v > 0 ? "+" : "") + v.toFixed(2)
            onMoved: v => root.set({ vertical_offset: v })
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Eye-line align")
            hint: qsTr("Rotate each crop so the eyes sit level")
            on: root.crop.eye_line_align
            onToggledTo: on => root.set({ eye_line_align: on })
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Padding")

        ColorRow {
            Layout.fillWidth: true
            title: qsTr("Fill colour")
            rgba: root.crop.fill_color
            onPicked: rgba => root.set({ fill_color: rgba })
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Used where a crop runs past the edge of the photo.")
            wrapMode: Text.Wrap
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("View and import")

        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Show crop overlay")
            hint: qsTr("Outline each face's crop on the photo")
            on: UiState.showCropOverlay
            onToggledTo: on => UiState.showCropOverlay = on
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Auto-orient via EXIF")
            hint: qsTr("Turn photos upright by their camera orientation; applies to photos added from now on")
            on: root.crop.auto_orient_exif
            onToggledTo: on => root.set({ auto_orient_exif: on })
        }
    }
}
