//! Regenerate the embedded PTX using NVIDIA CUDA Rust on Linux or rustc NVPTX on Windows.
//!
//! Run from the repository root: zig run tools/cuda/build.zig -- [--platform windows|linux] [--check]
const std = @import("std");
const Io = std.Io;

const kernels_path = "tools/cuda/kernels.rs";

fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn readFile(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    return Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
}

fn fingerprint(gpa: std.mem.Allocator, io: Io, platform: []const u8) ![64]u8 {
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    const project = try std.fmt.allocPrint(gpa, "tools/cuda/{s}", .{platform});
    var paths: std.ArrayList([]const u8) = .empty;
    try paths.append(gpa, kernels_path);
    for ([_][]const u8{ "Cargo.toml", "Cargo.lock", "rust-toolchain.toml" }) |name|
        try paths.append(gpa, try std.fs.path.join(gpa, &.{ project, name }));

    var src: std.ArrayList([]const u8) = .empty;
    var dir = try Io.Dir.cwd().openDir(io, try std.fs.path.join(gpa, &.{ project, "src" }), .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".rs"))
            try src.append(gpa, try gpa.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, src.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    for (src.items) |name| try paths.append(gpa, try std.fs.path.join(gpa, &.{ project, "src", name }));

    for (paths.items) |path| {
        const bytes = try readFile(gpa, io, path);
        var start: usize = 0;
        while (std.mem.indexOfPos(u8, bytes, start, "\r\n")) |i| { // CRLF -> LF, as git may check out either
            sha.update(bytes[start..i]);
            start = i + 1;
        }
        sha.update(bytes[start..]);
    }
    return std.fmt.bytesToHex(sha.finalResult(), .lower);
}

/// Names of every `.entry name(` in the PTX.
fn entryNames(gpa: std.mem.Allocator, ptx: []const u8) ![][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, ptx, pos, ".entry")) |i| {
        pos = i + ".entry".len;
        var j = pos;
        while (j < ptx.len and std.ascii.isWhitespace(ptx[j])) j += 1;
        if (j == pos) continue;
        const start = j;
        while (j < ptx.len and isWord(ptx[j])) j += 1;
        const end = j;
        while (j < ptx.len and std.ascii.isWhitespace(ptx[j])) j += 1;
        if (end > start and j < ptx.len and ptx[j] == '(') try names.append(gpa, ptx[start..end]);
    }
    return names.toOwnedSlice(gpa);
}

/// Names of every `fn fcs_*(` in the kernel source.
fn kernelNames(gpa: std.mem.Allocator, source: []const u8) ![][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, source, pos, "fn fcs_")) |i| {
        pos = i + 3;
        var j = pos;
        while (j < source.len and isWord(source[j])) j += 1;
        if (j < source.len and source[j] == '(') try names.append(gpa, source[pos..j]);
    }
    return names.toOwnedSlice(gpa);
}

/// Replace `word` wherever it appears as a whole identifier.
fn replaceWord(gpa: std.mem.Allocator, text: []const u8, word: []const u8, with: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, word)) |i| {
        const end = i + word.len;
        const bounded = (i == 0 or !isWord(text[i - 1])) and (end == text.len or !isWord(text[end]));
        try out.appendSlice(gpa, text[pos..i]);
        try out.appendSlice(gpa, if (bounded) with else word);
        pos = end;
    }
    try out.appendSlice(gpa, text[pos..]);
    return out.toOwnedSlice(gpa);
}

fn run(gpa: std.mem.Allocator, io: Io, argv: []const []const u8, cwd: []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv, .cwd = .{ .path = cwd } });
    const term = try child.wait(io);
    if (!term.success()) return error.CommandFailed;
    _ = gpa;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;

    var platform: []const u8 = if (@import("builtin").os.tag == .windows) "windows" else "linux";
    var check = false;
    const args = try init.minimal.args.toSlice(gpa);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--check")) {
            check = true;
        } else if (std.mem.eql(u8, args[i], "--platform") and i + 1 < args.len) {
            i += 1;
            platform = args[i];
        } else return error.BadArguments;
    }
    if (!std.mem.eql(u8, platform, "windows") and !std.mem.eql(u8, platform, "linux")) return error.BadArguments;

    var out_buf: [512]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &out_buf);
    const w = &stdout.interface;

    if (check) {
        for ([_][]const u8{ "windows", "linux" }) |p| {
            const artifact = try readFile(gpa, io, try std.fmt.allocPrint(gpa, "fcs-utils/src/cuda/{s}.ptx", .{p}));
            const want = try std.fmt.allocPrint(gpa, "// Source SHA256: {s}", .{try fingerprint(gpa, io, p)});
            if (std.mem.indexOf(u8, artifact, want) == null) {
                std.debug.print("Regenerate {s} PTX with tools/cuda/build.py\n", .{p});
                std.process.exit(1);
            }
        }
        try w.writeAll("Embedded CUDA artifacts match their Rust sources\n");
        try w.flush();
        return;
    }

    const project = try std.fmt.allocPrint(gpa, "tools/cuda/{s}", .{platform});
    var raw: []u8 = undefined;
    if (std.mem.eql(u8, platform, "windows")) {
        try run(gpa, io, &.{
            "cargo",               "+nightly-2026-08-24",                                  "rustc",
            "--manifest-path",     try std.fs.path.join(gpa, &.{ project, "Cargo.toml" }), "--target-dir",
            "target/cuda-kernels", "--target",                                             "nvptx64-nvidia-cuda",
            "-Z",                  "build-std=core",                                       "--release",
            "--",                  "--emit=asm",
        }, ".");
        const dir_path = "target/cuda-kernels/nvptx64-nvidia-cuda/release";
        var dir = try Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var walker = try dir.walk(gpa);
        var found: ?[]const u8 = null;
        while (try walker.next(io)) |entry| {
            const base = std.fs.path.basename(entry.path);
            if (std.mem.startsWith(u8, base, "fcs_cuda_kernels") and std.mem.endsWith(u8, base, ".s")) {
                if (found != null) return error.AmbiguousPtx;
                found = try std.fs.path.join(gpa, &.{ dir_path, entry.path });
            }
        }
        raw = try readFile(gpa, io, found orelse return error.NoPtx);
    } else {
        try run(gpa, io, &.{ "cargo", "oxide", "build", "--arch", "sm_80" }, project);
        raw = try readFile(gpa, io, try std.fs.path.join(gpa, &.{ project, "fcs_cuda_kernels.ptx" }));
    }

    var ptx: []u8 = raw;
    if (std.mem.indexOf(u8, ptx, ".extern .func") != null) {
        std.debug.print("PTX must be self-contained; found an unresolved device function\n", .{});
        std.process.exit(1);
    }
    const entries = try entryNames(gpa, ptx);
    for (try kernelNames(gpa, try readFile(gpa, io, kernels_path))) |name| {
        var matched: ?[]const u8 = null;
        for (entries) |entry| {
            const suffixed = entry.len > name.len and std.mem.endsWith(u8, entry, name) and entry[entry.len - name.len - 1] == '_';
            if (std.mem.eql(u8, entry, name) or suffixed) {
                if (matched != null) return error.AmbiguousEntry;
                matched = entry;
            }
        }
        ptx = try replaceWord(gpa, ptx, matched orelse return error.MissingEntry, name);
    }

    const output = try std.fmt.allocPrint(gpa, "fcs-utils/src/cuda/{s}.ptx", .{platform});
    try Io.Dir.cwd().createDirPath(io, "fcs-utils/src/cuda");
    const text = try std.fmt.allocPrint(
        gpa,
        "// Generated by tools/cuda/build.zig; edit tools/cuda/kernels.rs.\n// Source SHA256: {s}\n{s}",
        .{ try fingerprint(gpa, io, platform), ptx },
    );
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = output, .data = text });
    try w.print("{s}\n", .{output});
    try w.flush();
}
