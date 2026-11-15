import QtQuick
import QtQuick.Shapes

// Where one face's crop falls on the photo, traced in the crop shape. Decorative only:
// clicks go through to the face box.
Shape {
    id: root

    // A Studio.faces entry; its `crop` is { x, y, w, h } in fractions of the photo.
    required property var modelData
    required property real frameWidth
    required property real frameHeight
    readonly property var points: {
        const pts = Studio.shapeOutline.map(p => Qt.point(p[0] * width, p[1] * height))
        return pts.length > 0 ? pts.concat([pts[0]]) : pts
    }

    x: modelData.crop.x * frameWidth
    y: modelData.crop.y * frameHeight
    width: modelData.crop.w * frameWidth
    height: modelData.crop.h * frameHeight
    preferredRendererType: Shape.CurveRenderer
    Accessible.ignored: true

    // A dark backing keeps the line readable on light and dark photos alike.
    ShapePath {
        strokeColor: Qt.alpha(Theme.bg, 0.7)
        strokeWidth: 3.5
        fillColor: "transparent"
        joinStyle: ShapePath.RoundJoin
        PathPolyline { path: root.points }
    }
    ShapePath {
        strokeColor: root.modelData.selected ? Theme.ink : Qt.alpha(Theme.ink, 0.35)
        strokeWidth: 1.5
        strokeStyle: ShapePath.DashLine
        dashPattern: [5, 4]
        fillColor: "transparent"
        joinStyle: ShapePath.RoundJoin
        PathPolyline { path: root.points }
    }
}
