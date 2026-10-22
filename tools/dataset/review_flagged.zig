//! Contact sheet of eye labels whose left/right order is doubtful, for checking by eye.
//!
//!   zig run tools/dataset/review_flagged.zig -- <data_dir> [out.html]
//!
//! Two patterns need a human: clicks that arrived right-to-left on screen, and pairs whose eyes
//! sit nearly in vertical line. Neither is wrong by itself -- a face rolled past 90 degrees
//! legitimately reverses, and a steeply tilted one legitimately stacks -- but neither can be told
//! from a misclick by arithmetic, and getting it wrong swaps a face's eyes in training.
//!
//! Faces already carrying `upside_down` or `steep_roll` are left out, so this shows only what has
//! not been judged yet. Tiles are cut from the local JPEGs with CSS, so no image library is
//! needed; open the file locally. Green 1 is the first click, orange 2 the second.
const std = @import("std");
const u = @import("util.zig");

const TILE = 200;
const STEEP = 0.08; // horizontal eye separation, as a fraction of box width

const Flag = struct { record: std.json.Value, why: []const u8 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) return error.Usage;
    const data = try std.Io.Dir.cwd().realPathFileAlloc(io, args[1], gpa);
    const out_path = if (args.len > 2) args[2] else try std.fs.path.join(gpa, &.{ data, "review_flagged.html" });
    const images = try std.fs.path.join(gpa, &.{ data, "images" });

    const last = try u.lastWins(gpa, io, try std.fs.path.join(gpa, &.{ data, "eye_labels.jsonl" }), "id");
    var flagged: std.ArrayList(Flag) = .empty;
    for (last.values()) |record| {
        if (u.truthy(u.optField(record, "skipped")) or u.optField(record, "viewer_left") == null) continue;
        if (u.truthy(u.optField(record, "upside_down")) or u.truthy(u.optField(record, "steep_roll"))) continue; // already judged
        const left = u.field(record, "viewer_left").array.items;
        const right = u.field(record, "viewer_right").array.items;
        const box = u.field(record, "box").array.items;
        var width = u.num(box[2]) - u.num(box[0]);
        if (width == 0) width = 1e-9;
        if (u.num(left[0]) >= u.num(right[0])) {
            try flagged.append(gpa, .{ .record = record, .why = "clicked right-to-left" });
        } else if (@abs(u.num(right[0]) - u.num(left[0])) / width < STEEP) {
            try flagged.append(gpa, .{ .record = record, .why = "eyes nearly in vertical line" });
        }
    }

    var out: std.Io.Writer.Allocating = .init(gpa);
    const w = &out.writer;
    try w.writeAll("<!doctype html><meta charset=utf-8><title>Eye labels to check</title><style>" ++
        "body{font:13px system-ui;margin:24px;background:#111;color:#eee}" ++
        "section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:200px}" ++
        ".t{position:relative;width:200px;height:200px;background-repeat:no-repeat;" ++
        "border-radius:4px}" ++
        ".d{position:absolute;width:10px;height:10px;border-radius:50%;font-size:8px;" ++
        "color:#000;text-align:center;line-height:10px;font-weight:700}" ++
        "figcaption{color:#9ad;text-align:center;font-size:11px}h1{font-size:18px}" ++
        "p{color:#aaa;max-width:60em}</style>" ++
        "<h1>Click order to check</h1><p>Green <b>1</b> was the first click, orange <b>2</b> the " ++
        "second, and the order should run left to right across the face <i>as you see it</i>. " ++
        "A face rolled past vertical reverses that legitimately, which is the judgement needed " ++
        "here: upside down, or a slip?</p><section>");
    for (flagged.items) |f| {
        const record = f.record;
        const stem = u.str(u.field(record, "image"));
        const jpg = try std.fmt.allocPrint(gpa, "{s}{c}{s}.jpg", .{ images, std.fs.path.sep, stem });
        const size = u.jpegSizeOfFile(gpa, io, jpg) orelse continue;
        const iw: f64 = @floatFromInt(size[0]);
        const ih: f64 = @floatFromInt(size[1]);
        const box = u.field(record, "box").array.items;
        const x0, const y0, const x1, const y1 = .{ u.num(box[0]), u.num(box[1]), u.num(box[2]), u.num(box[3]) };
        const bw = (x1 - x0) * iw;
        const bh = (y1 - y0) * ih;
        const side = @max(bw, bh) * 1.9;
        const sx = x0 * iw + bw / 2 - side / 2;
        const sy = y0 * ih + bh / 2 - side / 2;
        const k = TILE / side;
        try w.print("<figure><div class=\"t\" style=\"background-image:url({s});background-size:{d:.1}px {d:.1}px;background-position:{d:.1}px {d:.1}px\">", .{ try u.asUri(gpa, jpg), iw * k, ih * k, -sx * k, -sy * k });
        for ([_][]const u8{ "viewer_left", "viewer_right" }, 0..) |name, index| {
            const p = u.field(record, name).array.items;
            try w.print("<div class=\"d\" style=\"left:{d:.1}px;top:{d:.1}px;background:{s}\">{d}</div>", .{ (u.num(p[0]) * iw - sx) * k - 5, (u.num(p[1]) * ih - sy) * k - 5, if (index == 0) "#4f8" else "#fa4", index + 1 });
        }
        const split = if (u.optField(record, "split")) |s| s.string else "first pass";
        try w.print("</div><figcaption>{s}<br>{s}<br>{s}</figcaption></figure>", .{ stem[0..@min(8, stem.len)], f.why, split });
    }
    try w.writeAll("</section>");
    try u.writeFile(io, out_path, out.written());
    var so: u.Stdout = undefined;
    so.init(io);
    try so.w().print("{d} faces to review -> {s}\n", .{ flagged.items.len, out_path });
    try so.w().flush();
}
