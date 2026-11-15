import QtQuick
import QtQuick.Layouts

InspectorPage {
    id: root

    readonly property var enhance: Studio.settings.enhance
    function set(fields) { Studio.update({ enhance: fields }) }
    function signed(v, digits) { return (v > 0 ? "+" : "") + v.toFixed(digits) }

    Section {
        Layout.fillWidth: true
        title: qsTr("Automatic")

        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Auto colour")
            hint: qsTr("Balance white and levels before the adjustments below")
            on: root.enhance.auto_color
            onToggledTo: on => root.set({ auto_color: on })
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Red-eye removal")
            on: root.enhance.red_eye_removal
            onToggledTo: on => root.set({ red_eye_removal: on })
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Background blur")
            hint: qsTr("Soften everything around the face")
            on: root.enhance.background_blur
            onToggledTo: on => root.set({ background_blur: on })
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Adjust")

        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Exposure")
            from: -3
            to: 3
            stepSize: 0.05
            value: root.enhance.exposure_stops
            format: v => qsTr("%1 EV").arg(root.signed(v, 2))
            onMoved: v => root.set({ exposure_stops: v })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Brightness")
            from: -100
            to: 100
            stepSize: 1
            value: root.enhance.brightness
            format: v => root.signed(v, 0)
            onMoved: v => root.set({ brightness: v })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Contrast")
            from: -2
            to: 2
            value: root.enhance.contrast
            // A multiplier: 1 leaves the image alone.
            format: v => "×" + v.toFixed(2)
            onMoved: v => root.set({ contrast: v })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Saturation")
            to: 3
            value: root.enhance.saturation
            format: v => "×" + v.toFixed(2)
            onMoved: v => root.set({ saturation: v })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Sharpness")
            to: 4
            value: root.enhance.sharpness
            onMoved: v => root.set({ sharpness: v })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Skin smoothing")
            to: 100
            value: root.enhance.skin_smooth * 100
            format: v => Math.round(v) + "%"
            onMoved: v => root.set({ skin_smooth: v / 100 })
        }
    }

    AppButton {
        Layout.alignment: Qt.AlignLeft
        variant: "ghost"
        text: qsTr("Restore defaults")
        onClicked: Studio.restoreEnhance()
    }
}
