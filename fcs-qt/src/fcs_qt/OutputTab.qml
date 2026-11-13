import QtQuick
import QtQuick.Layouts

InspectorPage {
    id: root

    readonly property var crop: Studio.settings.crop
    readonly property var rules: crop.quality_rules
    readonly property var gpu: Studio.settings.gpu
    function set(fields) { Studio.update({ crop: fields }) }
    function setRules(fields) { set({ quality_rules: fields }) }

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

    Section {
        Layout.fillWidth: true
        title: qsTr("Quality rules")

        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Only the best face per photo")
            hint: qsTr("Export the sharpest selected face of each photo and leave the rest")
            on: root.rules.auto_select_best_face
            onToggledTo: on => root.setRules({ auto_select_best_face: on })
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Skip photos without a sharp face")
            hint: qsTr("Leave out every face of a photo whose best crop is not high quality")
            on: root.rules.auto_skip_no_high_quality
            onToggledTo: on => root.setRules({ auto_skip_no_high_quality: on })
        }
        ToggleRow {
            Layout.fillWidth: true
            text: qsTr("Add quality to file names")
            hint: qsTr("Name crops …_highq, …_medq or …_lowq")
            on: root.rules.quality_suffix
            onToggledTo: on => root.setRules({ quality_suffix: on })
        }
        Text {
            Layout.topMargin: 4
            text: qsTr("Minimum quality")
            color: Theme.ink2
            font.pixelSize: Theme.body - 1
        }
        Segmented {
            Layout.fillWidth: true
            value: root.rules.min_quality ?? "any"
            options: [
                { value: "any", label: qsTr("Any") },
                { value: "medium", label: qsTr("Medium") },
                { value: "high", label: qsTr("High") }
            ]
            onActivated: value => root.setRules({ min_quality: value === "any" ? null : value })
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Detection")

        SliderRow {
            Layout.fillWidth: true
            title: qsTr("Confidence floor")
            from: 0.1
            to: 0.95
            stepSize: 0.01
            commitOnRelease: true
            value: Studio.settings.detection.confidence
            onMoved: v => Studio.update({ detection: { confidence: v } })
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Faces scored below this are ignored. Changing it finds the faces in every photo again, so hand edits are lost.")
            wrapMode: Text.Wrap
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Processing")
        detail: Studio.switching ? qsTr("switching…") : ""

        Segmented {
            Layout.fillWidth: true
            enabled: !Studio.switching
            value: !root.gpu.enabled ? "cpu" : root.gpu.experimental_cuda ? "cuda" : "gpu"
            options: [
                { value: "gpu", label: qsTr("GPU (WGSL)"), tip: qsTr("WGSL shaders on the GPU, the usual backend") },
                { value: "cuda", label: qsTr("CUDA"), enabled: Studio.cudaAvailable,
                  tip: Studio.cudaAvailable ? qsTr("Experimental CUDA Rust on %1").arg(Studio.cudaStatus)
                     : Studio.cudaStatus !== "" ? Studio.cudaStatus : qsTr("Checking for CUDA…") },
                { value: "cpu", label: qsTr("CPU"), tip: qsTr("No GPU: the built-in CPU graph") }
            ]
            onActivated: value => {
                if (value === "cuda")
                    cudaDialog.open()
                else
                    Studio.update({ gpu: { enabled: value === "gpu", experimental_cuda: false } })
            }
        }
        Text {
            Layout.fillWidth: true
            text: Studio.switching ? Studio.engineLabel
                : Studio.ready ? qsTr("Running on %1").arg(Studio.engineLabel) : ""
            wrapMode: Text.Wrap
            color: Theme.ink2
            font.pixelSize: Theme.caption
        }
        Text {
            Layout.fillWidth: true
            text: Studio.cudaStatus === "" ? qsTr("Checking for CUDA…")
                : Studio.cudaAvailable ? qsTr("CUDA device: %1").arg(Studio.cudaStatus)
                : qsTr("CUDA unavailable: %1").arg(Studio.cudaStatus)
            wrapMode: Text.Wrap
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
        ToggleRow {
            Layout.fillWidth: true
            enabled: root.gpu.enabled && !Studio.switching
            text: qsTr("Respect environment overrides")
            hint: qsTr("Let GPU environment variables such as WGPU_POWER_PREF choose the adapter")
            on: root.gpu.respect_env
            onToggledTo: on => Studio.update({ gpu: { respect_env: on } })
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Switching finds the faces in every photo again, so hand edits are lost.")
            wrapMode: Text.Wrap
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }

        ChoiceDialog {
            id: cudaDialog
            title: qsTr("Enable experimental CUDA Rust?")
            message: Studio.cudaWarning
            choices: [
                { value: "enable", label: qsTr("Enable CUDA Rust"), variant: "primary" },
                { value: "cancel", label: qsTr("Cancel"), variant: "ghost" }
            ]
            onChosen: value => {
                if (value === "enable")
                    Studio.update({ gpu: { enabled: true, experimental_cuda: true } })
            }
        }
    }

    Section {
        Layout.fillWidth: true
        title: qsTr("Report")

        Segmented {
            Layout.fillWidth: true
            value: UiState.report
            options: [
                { value: "none", label: qsTr("None") },
                { value: "csv", label: "CSV" },
                { value: "json", label: "JSON" }
            ]
            onActivated: value => UiState.report = value
        }
        Text {
            Layout.fillWidth: true
            text: qsTr("Saved beside the crops as face-crop-report, one line per photo.")
            wrapMode: Text.Wrap
            color: Theme.ink3
            font.pixelSize: Theme.caption
        }
    }
}
