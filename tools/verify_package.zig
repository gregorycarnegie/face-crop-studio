//! Exercise a packaged CLI, its models, and eye-aligned crops without ONNX Runtime.
//!
//! Usage: zig run tools/verify_package.zig -- <package-root> <cli-path> <sample-image>
//! Paths are resolved before running from an empty temporary working directory, so
//! models or settings in the checkout cannot accidentally rescue a broken package.
const std = @import("std");
const Io = std.Io;
const json = std.json;

fn fail(comptime fmt: []const u8, args: anytype) error{Failed} {
    std.debug.print(fmt ++ "\n", args);
    return error.Failed;
}

fn isInside(package: []const u8, path: []const u8) bool {
    return path.len > package.len and std.mem.startsWith(u8, path, package) and std.fs.path.isSep(path[package.len]);
}

fn hasDetectorLine(log: []const u8) bool {
    var lines = std.mem.splitScalar(u8, log, '\n');
    while (lines.next()) |line| {
        const at = std.mem.indexOf(u8, line, "Detector: ") orelse continue;
        const rest = line[at + "Detector: ".len ..];
        if (std.mem.indexOf(u8, rest, " on wgsl-gpu") != null or std.mem.indexOf(u8, rest, " on cpu-graph") != null) return true;
    }
    return false;
}

fn hasPng(gpa: std.mem.Allocator, io: Io, dir_path: []const u8) !bool {
    var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    while (try walker.next(io)) |e| {
        if (e.kind == .file and std.mem.endsWith(u8, e.basename, ".png")) return true;
    }
    return false;
}

/// `landmarks[:2]` of every detection; each eye is a JSON array or null.
fn eye(det: json.Value, i: usize) ?[2]f64 {
    const marks = det.object.get("landmarks").?.array.items;
    if (i >= marks.len or marks[i] != .array or marks[i].array.items.len < 2) return null;
    const xy = marks[i].array.items;
    return .{ num(xy[0]), num(xy[1]) };
}

fn num(v: json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => std.math.nan(f64),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 4) return fail("usage: verify_package <package-root> <cli-path> <sample-image>", .{});

    const cwd = Io.Dir.cwd();
    const package = try cwd.realPathFileAlloc(io, args[1], gpa);
    const cli = try cwd.realPathFileAlloc(io, args[2], gpa);
    const sample = try cwd.realPathFileAlloc(io, args[3], gpa);
    for ([_][]const u8{ cli, sample }) |path| {
        if (!isInside(package, path)) return fail("packaged file missing or outside package: {s}", .{path});
    }
    var pkg_dir = try cwd.openDir(io, package, .{ .iterate = true });
    defer pkg_dir.close(io);
    var walker = try pkg_dir.walk(gpa);
    while (try walker.next(io)) |e| {
        if (std.ascii.findIgnoreCase(e.basename, "onnxruntime") != null)
            return fail("ONNX Runtime must not be packaged: {s}", .{e.path});
    }

    var suffix: [8]u8 = undefined;
    io.random(&suffix);
    const tmp_root = init.environ_map.get("TEMP") orelse init.environ_map.get("TMPDIR") orelse "/tmp";
    const work = try std.fmt.allocPrint(gpa, "{s}{c}fcs-package-{x}", .{ tmp_root, std.fs.path.sep, suffix });
    try cwd.createDirPath(io, work);
    defer cwd.deleteTree(io, work) catch {};

    var env = try init.environ_map.clone(gpa);
    try env.put("RUST_LOG", "info");
    try env.put("ORT_DYLIB_PATH", try std.fmt.allocPrint(gpa, "{s}{c}absent-runtime", .{ work, std.fs.path.sep }));

    var results: [2][]const json.Value = undefined;
    for ([_]bool{ false, true }, 0..) |aligned, n| {
        const output = try std.fmt.allocPrint(gpa, "{s}{c}detections-{}.json", .{ work, std.fs.path.sep, aligned });
        const crops = try std.fmt.allocPrint(gpa, "{s}{c}crops", .{ work, std.fs.path.sep });
        const plain = [_][]const u8{ cli, "--input", sample, "--json", output };
        const extra = [_][]const u8{ "--crop", "--eye-line-align", "--output-dir", crops, "--output-format", "png", "--auto-detect-format=false" };
        const argv = if (aligned) try std.mem.concat(gpa, []const u8, &.{ &plain, &extra }) else @as([]const []const u8, &plain);

        // ponytail: no timeout; the CI step's own timeout is the backstop. Add `.timeout` to RunOptions if it must be per-run.
        const run = try std.process.run(gpa, io, .{ .argv = argv, .cwd = .{ .path = work }, .environ_map = &env });
        const log = try std.mem.concat(gpa, u8, &.{ run.stdout, run.stderr });
        std.debug.print("{s}\n", .{log});
        if (!run.term.success()) return fail("CLI failed: {any}", .{run.term});
        if (!hasDetectorLine(log)) return fail("packaged detector did not select a built-in engine", .{});
        if (aligned and std.mem.indexOf(u8, log, "eye refiner on the built-in CPU graph") == null)
            return fail("packaged eye-refiner model did not load", .{});

        const text = try cwd.readFileAlloc(io, output, gpa, .unlimited);
        const records = (try json.parseFromSliceLeaky(json.Value, gpa, text, .{})).array.items;
        if (records.len != 1) return fail("sample must produce at least one detection", .{});
        const dets = records[0].object.get("detections").?.array.items;
        if (dets.len == 0) return fail("sample must produce at least one detection", .{});
        results[n] = dets;
    }

    const before, const after = results;
    if (before.len != after.len) return fail("eye alignment changed the detection count", .{});
    var changed = false;
    for (before, after) |raw, refined| {
        for (0..2) |i| {
            const e = eye(refined, i) orelse return fail("refinement did not produce finite eye coordinates", .{});
            if (!std.math.isFinite(e[0]) or !std.math.isFinite(e[1])) return fail("refinement did not produce finite eye coordinates", .{});
            const o = eye(raw, i) orelse {
                changed = true;
                continue;
            };
            changed = changed or @abs(o[0] - e[0]) > 1e-4 or @abs(o[1] - e[1]) > 1e-4;
        }
    }
    const crops = try std.fmt.allocPrint(gpa, "{s}{c}crops", .{ work, std.fs.path.sep });
    if (!changed or !try hasPng(gpa, io, crops)) return fail("expected refined eyes and an exported PNG crop", .{});
    std.debug.print("PACKAGE OK: {} faces, refined eyes and crops; no ONNX Runtime\n", .{after.len});
}
