import QtQuick
import QtQuick.Shapes

Item {
    id: root

    property string path
    property color color: Theme.ink
    property real size: 18

    implicitWidth: size
    implicitHeight: size
    Accessible.ignored: true

    Shape {
        anchors.centerIn: parent
        width: 24
        height: 24
        scale: root.size / 24
        preferredRendererType: Shape.CurveRenderer

        ShapePath {
            strokeColor: root.color
            strokeWidth: 1.8
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin
            PathSvg { path: root.path }
        }
    }
}
