import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import ".."
import "../components" as Components

Item {
    id: notesApp
    anchors.fill: parent

    property int selectedNote: 0
    property string noteFilter: ""

    property var notes: [
        { id: 1, title: "Meeting Notes", content: "NeuralOS v3.4 Release Planning\n\n- Window button redesign complete\n- Task Manager implemented\n- Calendar and Notes apps added\n- Performance optimization pending\n\nAction items:\n1. Run full test suite\n2. Update documentation\n3. Prepare release notes", date: "2026-02-07", pinned: true },
        { id: 2, title: "Server Setup", content: "Production Server Checklist:\n\n[ ] Install NeuralOS base image\n[ ] Configure NPU drivers\n[ ] Deploy AI runtime\n[ ] Setup firewall rules\n[ ] Enable SSH access\n[ ] Configure monitoring\n[ ] Run security audit", date: "2026-02-06", pinned: true },
        { id: 3, title: "Project Ideas", content: "Future NeuralOS Features:\n\n1. AI-powered file search\n2. Voice command integration\n3. Smart notification grouping\n4. Predictive app launching\n5. Neural theme generation\n6. Automated system tuning", date: "2026-02-05", pinned: false },
        { id: 4, title: "Quick Reminders", content: "- Update NPU drivers to v2.1\n- Run system backup tonight\n- Review pull request #847\n- Test dark/light mode transitions\n- Check memory leak in ai-runtime", date: "2026-02-04", pinned: false }
    ]

    property int nextId: 5

    function getFilteredNotes() {
        var list = notes.slice()
        if (noteFilter !== "") {
            list = list.filter(function(n) {
                return n.title.toLowerCase().indexOf(noteFilter.toLowerCase()) !== -1 ||
                       n.content.toLowerCase().indexOf(noteFilter.toLowerCase()) !== -1
            })
        }
        list.sort(function(a, b) {
            if (a.pinned !== b.pinned) return a.pinned ? -1 : 1
            return b.id - a.id
        })
        return list
    }

    function addNote() {
        var n = notes.slice()
        n.push({ id: nextId, title: "New Note", content: "", date: Qt.formatDate(new Date(), "yyyy-MM-dd"), pinned: false })
        nextId++
        notes = n
        selectedNote = notes.length - 1
    }

    function deleteNote() {
        if (notes.length <= 1) return
        var filtered = getFilteredNotes()
        if (selectedNote >= filtered.length) return
        var targetId = filtered[selectedNote].id
        notes = notes.filter(function(n) { return n.id !== targetId })
        if (selectedNote >= getFilteredNotes().length) selectedNote = Math.max(0, getFilteredNotes().length - 1)
    }

    function togglePin() {
        var filtered = getFilteredNotes()
        if (selectedNote >= filtered.length) return
        var targetId = filtered[selectedNote].id
        var n = notes.slice()
        for (var i = 0; i < n.length; i++) {
            if (n[i].id === targetId) {
                n[i] = Object.assign({}, n[i])
                n[i].pinned = !n[i].pinned
                break
            }
        }
        notes = n
    }

    function updateNoteContent(text) {
        var filtered = getFilteredNotes()
        if (selectedNote >= filtered.length) return
        var targetId = filtered[selectedNote].id
        var n = notes.slice()
        for (var i = 0; i < n.length; i++) {
            if (n[i].id === targetId) {
                n[i] = Object.assign({}, n[i])
                n[i].content = text
                n[i].date = Qt.formatDate(new Date(), "yyyy-MM-dd")
                break
            }
        }
        notes = n
    }

    function updateNoteTitle(text) {
        var filtered = getFilteredNotes()
        if (selectedNote >= filtered.length) return
        var targetId = filtered[selectedNote].id
        var n = notes.slice()
        for (var i = 0; i < n.length; i++) {
            if (n[i].id === targetId) {
                n[i] = Object.assign({}, n[i])
                n[i].title = text
                break
            }
        }
        notes = n
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.background

        RowLayout {
            anchors.fill: parent
            spacing: 0

            /* ─── Left: Note List ─── */
            Rectangle {
                Layout.preferredWidth: 240
                Layout.fillHeight: true
                color: Theme.surface

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 0

                    /* Header */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 48
                        color: "transparent"

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 10; spacing: 8

                            Text {
                                Layout.fillWidth: true; text: "Notes"
                                font.pixelSize: 16; font.bold: true; color: Theme.text
                            }

                            Rectangle {
                                width: 28; height: 28; radius: Theme.radiusSmall
                                color: newNoteMa.containsMouse ? Theme.surfaceAlt : "transparent"
                                Text { anchors.centerIn: parent; text: "+"; font.pixelSize: 18; color: Theme.primary }
                                MouseArea { id: newNoteMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: addNote() }
                            }
                        }
                    }

                    /* Search */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 32
                        Layout.leftMargin: 10; Layout.rightMargin: 10
                        radius: 14; color: Theme.surfaceAlt

                        RowLayout {
                            anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10; spacing: 6
                            Components.CanvasIcon { iconName: "search"; iconSize: 12; iconColor: Theme.textDim }
                            TextInput {
                                Layout.fillWidth: true; font.pixelSize: 12; color: Theme.text
                                clip: true; selectByMouse: true
                                onTextChanged: { noteFilter = text; selectedNote = 0 }
                                Text {
                                    visible: !parent.text && !parent.activeFocus
                                    text: "Search notes..."; color: Theme.textMuted; font.pixelSize: 12
                                }
                            }
                        }
                    }

                    Item { height: 8 }

                    /* Note List */
                    ListView {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        clip: true; model: getFilteredNotes(); spacing: 2

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded; width: 4
                            contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                        }

                        delegate: Rectangle {
                            width: ListView.view ? ListView.view.width - 8 : 0; height: 64
                            x: 4; radius: Theme.radiusSmall
                            color: index === selectedNote
                                ? Qt.rgba(Theme.primary.r, Theme.primary.g, Theme.primary.b, 0.12)
                                : noteItemMa.containsMouse ? Theme.hoverBg : "transparent"

                            ColumnLayout {
                                anchors.fill: parent; anchors.margins: 10; spacing: 3

                                RowLayout {
                                    Layout.fillWidth: true; spacing: 4
                                    Components.CanvasIcon {
                                        visible: modelData.pinned
                                        iconName: "star"; iconSize: 10; iconColor: Theme.warning
                                    }
                                    Text {
                                        Layout.fillWidth: true; text: modelData.title
                                        font.pixelSize: 12; font.weight: Font.DemiBold
                                        color: Theme.text; elide: Text.ElideRight
                                    }
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.content.replace(/\n/g, " ").substring(0, 60)
                                    font.pixelSize: 10; color: Theme.textDim; elide: Text.ElideRight
                                }

                                Text { text: modelData.date; font.pixelSize: 9; color: Theme.textMuted }
                            }

                            MouseArea {
                                id: noteItemMa; anchors.fill: parent
                                hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: selectedNote = index
                            }
                        }
                    }

                    /* Footer */
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 32
                        color: Theme.surfaceAlt

                        Text {
                            anchors.centerIn: parent
                            text: notes.length + " notes"
                            font.pixelSize: 10; color: Theme.textDim
                        }
                    }
                }
            }

            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: Theme.surfaceLight }

            /* ─── Right: Editor ─── */
            ColumnLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                spacing: 0

                /* Editor Toolbar */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 44
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; spacing: 6

                        TextInput {
                            Layout.fillWidth: true
                            text: getFilteredNotes().length > selectedNote ? getFilteredNotes()[selectedNote].title : ""
                            font.pixelSize: 16; font.bold: true; color: Theme.text
                            selectByMouse: true; selectionColor: Theme.primary
                            onTextChanged: if (activeFocus) updateNoteTitle(text)
                        }

                        Repeater {
                            model: [
                                { ico: "star", tip: "Pin", act: "pin" },
                                { ico: "trash", tip: "Delete", act: "delete" }
                            ]

                            Rectangle {
                                width: 30; height: 30; radius: Theme.radiusSmall
                                color: edBtnMa.containsMouse
                                    ? (modelData.act === "delete" ? Qt.rgba(Theme.error.r, Theme.error.g, Theme.error.b, 0.12) : Theme.surfaceAlt)
                                    : "transparent"

                                Components.CanvasIcon {
                                    anchors.centerIn: parent; iconName: modelData.ico; iconSize: 14
                                    iconColor: modelData.act === "delete" ? Theme.error : Theme.textDim
                                }

                                MouseArea {
                                    id: edBtnMa; anchors.fill: parent
                                    hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (modelData.act === "pin") togglePin()
                                        else if (modelData.act === "delete") deleteNote()
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.surfaceLight }

                /* Text Editor */
                Flickable {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    contentHeight: noteEdit.paintedHeight + 24; clip: true
                    flickableDirection: Flickable.VerticalFlick

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded; width: 4
                        contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.textMuted; opacity: 0.5 }
                    }

                    TextEdit {
                        id: noteEdit
                        width: parent.width; padding: 14
                        text: getFilteredNotes().length > selectedNote ? getFilteredNotes()[selectedNote].content : ""
                        font.pixelSize: 13; font.family: "monospace"
                        color: Theme.text; selectionColor: Theme.primary; selectedTextColor: "#FFFFFF"
                        wrapMode: TextEdit.Wrap; selectByMouse: true
                        onTextChanged: if (activeFocus) updateNoteContent(text)
                    }
                }

                /* Footer */
                Rectangle {
                    Layout.fillWidth: true; Layout.preferredHeight: 28
                    color: Theme.surface

                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14

                        Text {
                            text: {
                                var n = getFilteredNotes()
                                if (selectedNote < n.length) {
                                    var c = n[selectedNote].content
                                    return c.length + " characters \u2022 " + c.split(/\n/).length + " lines"
                                }
                                return ""
                            }
                            font.pixelSize: 10; color: Theme.textMuted
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: getFilteredNotes().length > selectedNote ? "Modified: " + getFilteredNotes()[selectedNote].date : ""
                            font.pixelSize: 10; color: Theme.textMuted
                        }
                    }
                }
            }
        }
    }
}
