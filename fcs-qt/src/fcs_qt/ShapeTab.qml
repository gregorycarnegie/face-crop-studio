pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

InspectorPage {
    id: root

    readonly property var crop: Studio.settings.crop
    readonly property var shape: crop.shape
    readonly property string kind: shape.kind
    readonly property string corner: shape.corner_style?.style ?? ""
    readonly property int sides: shape.sides ?? 6
    readonly property string label: {
        const entry = Studio.shapes.find(s => root.matches(s.shape))
        return entry ? entry.label : ""
    }

    function matches(s) {
        return s.kind === kind && (s.kind !== "polygon" || s.corner_style.style === corner)
    }
    // Merged into the current shape, so one parameter changes and the rest stay.
    function setShape(fields) { Studio.update({ crop: { shape: fields } }) }
    function setCorner(fields) { setShape({ corner_style: fields }) }
    function set(fields) { Studio.update({ crop: fields }) }
    function percent(v) { return Math.round(v) + "%" }
    function degrees(v) { return Math.round(v) + "°" }

    Section {
        Layout.fillWidth: true
        title: qsTr("Shape")
        detail: root.label

        GridLayout {
            Layout.fillWidth: true
            columns: 5
            rowSpacing: 8
            columnSpacing: 8

            Repeater {
                model: Studio.shapes
                delegate: ShapeChip {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    active: root.matches(modelData.shape)
                    onClicked: root.set({ shape: modelData.shape })
                }
            }
        }
    }

    Section {
        Layout.fillWidth: true
        visible: root.kind !== "rectangle" && root.kind !== "ellipse"
        title: qsTr("Shape settings")

        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "rounded_rectangle"
            title: qsTr("Corner radius")
            to: 50
            value: (root.shape.radius_pct ?? 0) * 100
            format: root.percent
            onMoved: v => root.setShape({ radius_pct: v / 100 })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "chamfered_rectangle"
            title: qsTr("Chamfer size")
            to: 50
            value: (root.shape.size_pct ?? 0) * 100
            format: root.percent
            onMoved: v => root.setShape({ size_pct: v / 100 })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "polygon" || root.kind === "koch_polygon"
            title: qsTr("Sides")
            from: 3
            to: 24
            stepSize: 1
            value: root.sides
            format: v => Math.round(v)
            onMoved: v => root.setShape({ sides: v })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "star"
            title: qsTr("Points")
            from: 3
            to: 24
            stepSize: 1
            value: root.shape.points ?? 5
            format: v => Math.round(v)
            onMoved: v => root.setShape({ points: v })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "star"
            title: qsTr("Inner radius")
            from: 10
            to: 90
            value: (root.shape.inner_radius_pct ?? 0.5) * 100
            format: root.percent
            onMoved: v => root.setShape({ inner_radius_pct: v / 100 })
        }
        // A polygon's corner can only round or bevel so far before the sides vanish.
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "polygon" && root.corner === "rounded"
            title: qsTr("Corner radius")
            to: 50 * Math.cos(Math.PI / root.sides)
            value: (root.shape.corner_style?.radius_pct ?? 0) * 100
            format: v => v.toFixed(1) + "%"
            onMoved: v => root.setCorner({ radius_pct: v / 100 })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "polygon" && root.corner === "chamfered"
            title: qsTr("Chamfer size")
            to: 50 * Math.sin(Math.PI / root.sides)
            value: (root.shape.corner_style?.size_pct ?? 0) * 100
            format: v => v.toFixed(1) + "%"
            onMoved: v => root.setCorner({ size_pct: v / 100 })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "polygon" && root.corner === "bezier"
            title: qsTr("Tension")
            to: 2
            value: root.shape.corner_style?.tension ?? 0.5
            onMoved: v => root.setCorner({ tension: v })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "polygon" || root.kind === "star" || root.kind === "koch_polygon"
            title: qsTr("Rotation")
            from: -180
            to: 180
            stepSize: 1
            value: root.shape.rotation_deg ?? 0
            format: root.degrees
            onMoved: v => root.setShape({ rotation_deg: v })
        }
        SliderRow {
            Layout.fillWidth: true
            visible: root.kind === "koch_polygon" || root.kind === "koch_rectangle"
            title: qsTr("Iterations")
            to: 5
            stepSize: 1
            value: root.shape.iterations ?? 3
            format: v => Math.round(v)
            onMoved: v => root.setShape({ iterations: v })
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Vignette")

        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Softness")
            to: 100
            value: root.crop.vignette_softness * 100
            format: root.percent
            onMoved: v => root.set({ vignette_softness: v / 100 })
        }
        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Intensity")
            to: 100
            value: root.crop.vignette_intensity * 100
            format: root.percent
            onMoved: v => root.set({ vignette_intensity: v / 100 })
        }
        ColorRow {
            Layout.fillWidth: true
            title: qsTr("Vignette colour")
            rgba: root.crop.vignette_color
            onPicked: rgba => root.set({ vignette_color: rgba })
        }
    }
}
