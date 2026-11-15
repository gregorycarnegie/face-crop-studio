.pragma library

// The first item under `item` (itself included) that passes `test`, searching children and
// a Flickable's content.
function descendant(item, test) {
    if (!item)
        return null
    if (test(item))
        return item
    const kids = Array.prototype.slice.call(item.children || [])
    if (item.contentItem && kids.indexOf(item.contentItem) < 0)
        kids.push(item.contentItem)
    for (const child of kids) {
        const found = descendant(child, test)
        if (found)
            return found
    }
    return null
}

// A queue row as Studio sends it (studio.rs: queue_row).
function row(name, extra) {
    return Object.assign({ "name": name, "thumb": "", "status": "done", "error": "",
                           "faces": 1, "selected": 1, "mapped": "" }, extra || {})
}
