import QtQuick
import QtQuick.Layouts

InspectorPage {
    id: root

    readonly property var crop: Studio.settings.crop
    function set(fields) { Studio.update({ crop: fields }) }

    Section {
        Layout.fillWidth: true
        title: qsTr("Format")

        Segmented {
            Layout.fillWidth: true
            value: root.crop.output_format
            options: [
                { value: "png", label: "PNG" },
                { value: "jpeg", label: "JPEG" },
                { value: "webp", label: "WebP" },
                { value: "avif", label: "AVIF" }
            ]
            onActivated: value => root.set({ output_format: value })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.crop.output_format === "jpeg"
            title: qsTr("JPEG quality")
            from: 50
            to: 100
            stepSize: 1
            value: root.crop.jpeg_quality
            format: v => Math.round(v)
            onMoved: v => root.set({ jpeg_quality: v })
        }
    }
}
