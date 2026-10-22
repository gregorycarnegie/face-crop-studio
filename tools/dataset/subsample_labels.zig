//! Make nested label variants, to measure how many eye points the landmark head needs.
//!
//!   zig run tools/dataset/subsample_labels.zig -- <labelv2.txt> <out_dir> [sizes...]   (default: 400 800 1600 all)
//!
//! Every variant keeps every image and every box. Only the *landmark* supervision shrinks: in the
//! 400-face variant, 400 faces keep their eye points and the rest are rewritten as `-1 -1 -1`,
//! which SCRFD's `RetinaFaceDataset` drops from the landmark loss while still training the box.
//! That isolates "how many eye labels" from "how much data", which a curve built by throwing away
//! whole images could not do.
//!
//! The selection is seeded and nested, so 400 is a subset of 800 is a subset of 1600. Without
//! that, a difference between two points on the curve could just be which faces got picked. The
//! shuffle reproduces Python's `random.Random(0).shuffle`, so the variants match earlier runs.
const std = @import("std");
const u = @import("util.zig");

const KEYPOINT_FIELDS = 15; // 5 landmarks x (x, y, weight)
const IGNORED = "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1";

fn hasPoints(line: []const u8) bool {
    var n: usize = 0;
    var any = false;
    var it = std.mem.tokenizeAny(u8, line, " \t\r");
    while (it.next()) |v| : (n += 1) {
        if (n >= 4 and !std.mem.eql(u8, v, "-1")) any = true;
    }
    return n >= 4 + KEYPOINT_FIELDS and any;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 3) return error.Usage;
    const source = args[1];
    const out_dir = args[2];
    const sizes: []const []const u8 = if (args.len > 3) @ptrCast(args[3..]) else &.{ "400", "800", "1600", "all" };

    const text = try u.readFile(gpa, io, source);
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| try lines.append(gpa, std.mem.trimEnd(u8, l, "\r"));
    if (lines.items.len > 0 and lines.items[lines.items.len - 1].len == 0) _ = lines.pop();

    var labelled: std.ArrayList(usize) = .empty;
    for (lines.items, 0..) |l, i| {
        if (!std.mem.startsWith(u8, l, "#") and hasPoints(l)) try labelled.append(gpa, i);
    }
    const order = try gpa.dupe(usize, labelled.items);
    var rng = u.PyRandom.init(0);
    rng.shuffle(usize, order);

    const stem = std.fs.path.stem(source);
    var so: u.Stdout = undefined;
    so.init(io);
    const w = so.w();
    try w.print("{s}: {d} lines, {d} faces with eye points\n", .{ std.fs.path.basename(source), lines.items.len, labelled.items.len });

    for (sizes) |size| {
        const is_all = std.mem.eql(u8, size, "all");
        const asked = if (is_all) order.len else try std.fmt.parseInt(usize, size, 10);
        const count = @min(asked, order.len);
        const keep = try gpa.alloc(bool, lines.items.len);
        @memset(keep, false);
        for (order[0..count]) |i| keep[i] = true;

        var out: std.Io.Writer.Allocating = .init(gpa);
        var kept: usize = 0;
        for (lines.items, 0..) |l, i| {
            if (std.mem.indexOfScalar(usize, labelled.items, i) != null and !keep[i]) {
                var parts = std.mem.tokenizeAny(u8, l, " \t");
                for (0..4) |n| try out.writer.print("{s}{s}", .{ if (n > 0) " " else "", parts.next().? });
                try out.writer.print(" {s}\n", .{IGNORED});
            } else {
                try out.writer.print("{s}\n", .{l});
                if (keep[i]) kept += 1;
            }
        }
        const name = try std.fmt.allocPrint(gpa, "{s}_{d}.txt", .{ stem, count });
        try u.writeFile(io, try std.fs.path.join(gpa, &.{ out_dir, name }), out.written());
        try w.print("  {s}: {d} faces keep their eye points, {d} rewritten as ignored\n", .{ name, kept, labelled.items.len - kept });
        if (!is_all and count < asked) try w.print("    (asked for {s}, only {d} labelled faces exist)\n", .{ size, order.len });
    }
    try w.flush();
}
