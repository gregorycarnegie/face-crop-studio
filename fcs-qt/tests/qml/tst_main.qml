import QtQuick
import QtTest
import fcs_qt

// The whole window, built against the mock: every screen must load without a QML error.
TestCase {
    name: "Main"

    Component {
        id: mainWindow
        Main {}
    }

    function test_the_window_builds_without_errors() {
        const window = createTemporaryObject(mainWindow, null)
        verify(window, "Main.qml failed to build; see the errors above")
        window.visible = true
        tryVerify(() => window.width > 0)
    }
}
