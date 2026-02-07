.pragma library

var _windows = {}
var _nextId = 1
var _topZ = 100
var _focusedId = -1

function openWindow(props) {
    if (props.singleton) {
        for (var key in _windows) {
            if (_windows[key].source === props.source) {
                focusWindow(parseInt(key))
                return parseInt(key)
            }
        }
    }

    var id = _nextId++
    _windows[id] = {
        id: id,
        source: props.source,
        title: props.title || "Untitled",
        icon: props.icon || "\u25A3",
        color: props.color || "#00D9FF",
        width: props.width || 800,
        height: props.height || 500,
        x: props.x !== undefined ? props.x : 80 + ((id % 6) * 40),
        y: props.y !== undefined ? props.y : 40 + ((id % 6) * 30),
        minimized: false,
        maximized: false,
        zOrder: ++_topZ,
        item: null
    }
    _focusedId = id
    return id
}

function closeWindow(id) {
    if (_windows[id] && _windows[id].item) {
        _windows[id].item.destroy()
    }
    delete _windows[id]
    if (_focusedId === id) _focusedId = -1
}

function focusWindow(id) {
    if (!_windows[id]) return _topZ
    _windows[id].zOrder = ++_topZ
    _focusedId = id
    if (_windows[id].item) {
        _windows[id].item.z = _windows[id].zOrder
    }
    return _windows[id].zOrder
}

function minimizeWindow(id) {
    if (!_windows[id]) return
    _windows[id].minimized = true
    if (_windows[id].item) _windows[id].item.visible = false
}

function restoreWindow(id) {
    if (!_windows[id]) return
    _windows[id].minimized = false
    if (_windows[id].item) _windows[id].item.visible = true
    focusWindow(id)
}

function toggleMaximize(id) {
    if (!_windows[id]) return
    _windows[id].maximized = !_windows[id].maximized
    return _windows[id].maximized
}

function getWindow(id) {
    return _windows[id] || null
}

function getWindowList() {
    var list = []
    for (var key in _windows) {
        var w = _windows[key]
        w.focused = (w.id === _focusedId)
        list.push(w)
    }
    return list.sort(function(a, b) { return a.id - b.id })
}

function getWindowCount() {
    var count = 0
    for (var key in _windows) count++
    return count
}

function getFocusedId() {
    return _focusedId
}

function isOpen(source) {
    for (var key in _windows) {
        if (_windows[key].source === source) return true
    }
    return false
}
