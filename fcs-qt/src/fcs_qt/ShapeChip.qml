import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Shapes

// One entry of the shape picker, drawn from the shape's real outline.
AbstractButton {
    id: control

    // { label, outline: [[x, y], ...] in 0..1, shape } from Studio.shapes
    required property var modelData
    property bool active

    implicitWidth: 52
    implicitHeight: 52
    hoverEnabled: true
    Accessible.role: Accessible.RadioButton
    Accessible.name: modelData.label
    Accessible.checked: active
    ToolTip.visible: hovered
    ToolTip.delay: 300
    ToolTip.text: modelData.label

    background: Rectangle {
        radius: Theme.radius
        color: control.active ? Qt.alpha(Theme.accent, 0.12) : control.hovered ? Theme.surface2 : Theme.surface
        border.width: control.visualFocus ? 2 : 1
        border.color: control.visualFocus ? Theme.focus : control.active ? Theme.accent : Theme.rule

        Behavior on color { ColorAnimation { duration: Theme.fast } }
    }

    contentItem: Item {
        Shape {
            id: glyph
            readonly property real side: 26
            anchors.centerIn: parent
            width: side
            height: side
            preferredRendererType: Shape.CurveRenderer

            ShapePath {
                strokeColor: control.active ? Theme.accent : Theme.ink2
                strokeWidth: 1.6
                fillColor: control.active ? Qt.alpha(Theme.accent, 0.18) : "transparent"
                joinStyle: ShapePath.RoundJoin
                PathPolyline {
                    path: {
                        const pts = control.modelData.outline.map(p => Qt.point(p[0] * glyph.side, p[1] * glyph.side))
                        return pts.length > 0 ? pts.concat([pts[0]]) : pts
                    }
                }
            }
        }
    }
}
