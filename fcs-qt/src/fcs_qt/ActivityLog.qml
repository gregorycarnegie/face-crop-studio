pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// What the app has done this session, newest at the bottom.
ListView {
    id: root

    clip: true
    model: Studio.log
    spacing: 2
    leftMargin: 12
    rightMargin: 12
    bottomMargin: 12
    boundsBehavior: Flickable.StopAtBounds
    Accessible.role: Accessible.List
    Accessible.name: qsTr("Activity")
    ScrollBar.vertical: ScrollBar {}

    // Follow new lines, unless the user has scrolled back to read.
    property bool following: true
    onMovementEnded: following = atYEnd
    onCountChanged: if (following) Qt.callLater(positionViewAtEnd)

    delegate: RowLayout {
        id: line

        required property var modelData

        width: ListView.view.width - ListView.view.leftMargin - ListView.view.rightMargin
        spacing: 8

        Text {
            Layout.alignment: Qt.AlignTop
            text: line.modelData.time
            color: Theme.ink3
            font.pixelSize: Theme.caption
            font.features: ({ "tnum": 1 })
        }
        Rectangle {
            Layout.alignment: Qt.AlignTop
            Layout.topMargin: 5
            Layout.preferredWidth: 6
            Layout.preferredHeight: 6
            radius: 3
            color: line.modelData.kind === "error" ? Theme.red
                 : line.modelData.kind === "warn" ? Theme.peach
                 : line.modelData.kind === "ok" ? Theme.green : Theme.rule2
            Accessible.ignored: true
        }
        Text {
            Layout.fillWidth: true
            text: line.modelData.text
            wrapMode: Text.Wrap
            color: line.modelData.kind === "error" ? Theme.red : Theme.ink2
            font.pixelSize: Theme.caption
            Accessible.role: Accessible.StaticText
            Accessible.name: text
        }
    }

    Text {
        anchors.centerIn: parent
        width: parent.width - 48
        visible: root.count === 0
        text: qsTr("Nothing has happened yet.")
        horizontalAlignment: Text.AlignHCenter
        color: Theme.ink3
        font.pixelSize: Theme.caption
    }
}
