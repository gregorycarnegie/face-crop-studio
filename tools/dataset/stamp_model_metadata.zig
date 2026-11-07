//! Stamp provenance into an ONNX model without decoding or rewriting its graph.
//!   zig run tools/dataset/stamp_model_metadata.zig -- scrfd|eye_refiner <model.onnx>
const std = @import("std");
const pb = @import("onnx.zig");
const u = @import("util.zig");
const Allocator = std.mem.Allocator;

const repo = "https://github.com/gregorycarnegie/face-crop-studio";
const author = "Face Crop Studio Contributors";
const license_url = "https://creativecommons.org/licenses/by/4.0/";
const producer = "Face Crop Studio";
pub const Model = enum { scrfd, eye_refiner };

const Info = struct {
    title: []const u8,
    version: u64,
    summary: []const u8,
    io: []const u8,
    training_data: []const u8,
    produced_by: []const u8,
};

fn info(model: Model) Info {
    return switch (model) {
        .scrfd => .{
            .title = "SCRFD-80k face detector",
            .version = 2,
            .summary = "SCRFD-500M face detector trained by Face Crop Studio on 80,000 CC BY 2.0 Open " ++
                "Images photographs (244,683 boxed faces), with eye landmarks only.",
            .io = "Input `input.1` [1, 3, 640, 640], RGB, (pixel - 127.5) / 128, the source " ++
                "letterboxed into the top-left. Outputs: nine tensors, three per stride 8/16/32 -- " ++
                "sigmoided scores [1, N, 1], box distances [1, N, 4] and keypoint distances " ++
                "[1, N, 10] in units of the stride. No NMS in the graph. Of the five keypoints only " ++
                "the two eyes were trained.",
            .training_data = "Open Images V7, 80,000 photographs, CC BY 2.0",
            .produced_by = "tools/dataset/export_scrfd.py",
        },
        .eye_refiner => .{
            .title = "Face Crop Studio eye refiner",
            .version = 1,
            .summary = "Refines a face detector's two eye points for eye-line alignment. Trained by Face " ++
                "Crop Studio from scratch on 1,988 hand-clicked eye pairs from CC BY 2.0 Open " ++
                "Images photographs.",
            .io = "Input `crop` [batch, 3, 112, 112], RGB, (pixel - 127.5) / 128: the face box " ++
                "expanded to max(w, h) * 1.25 about its centre. Output `eyes` [batch, 4]: two eye " ++
                "points as fractions of the crop side, the viewer's left eye first.",
            .training_data = "Open Images V7, 1,988 hand-clicked eye pairs, CC BY 2.0",
            .produced_by = "tools/dataset/train_eye_refiner.py, exported by export_eye_refiner.py",
        },
    };
}

const keys = [_][]const u8{ "title", "author", "source", "license", "license_url", "attribution", "training_data", "produced_by" };

pub fn stamp(gpa: Allocator, data: []const u8, model: Model) ![]const u8 {
    const meta = info(model);
    const graph = try (try pb.field(data, 7) orelse return error.MissingGraph).bytes();
    if (graph.len == 0 or (try pb.field(data, 1)) == null or (try pb.field(data, 8)) == null) return error.InvalidModel;
    var graph_fields: pb.Reader = .{ .data = graph };
    while (try graph_fields.next()) |_| {}

    var out: std.ArrayList(u8) = .empty;
    var r: pb.Reader = .{ .data = data };
    while (try r.next()) |f| {
        if (f.tag == 2 or f.tag == 3 or f.tag == 5 or f.tag == 6) continue;
        if (f.tag == 14) {
            const key = try pb.string(try f.bytes(), 1);
            var replaced = false;
            for (keys) |k| if (std.mem.eql(u8, k, key)) {
                replaced = true;
            };
            if (replaced) continue;
        }
        try out.appendSlice(gpa, f.raw);
    }
    const original_producer = try pb.string(data, 2);
    var version = try pb.string(data, 3);
    if (!std.mem.eql(u8, original_producer, producer)) {
        const exporter = std.mem.trim(u8, try std.fmt.allocPrint(gpa, "{s} {s}", .{ original_producer, version }), " \t\r\n");
        version = if (exporter.len > 0) try std.fmt.allocPrint(gpa, "exported with {s}", .{exporter}) else "";
    }
    const credit = try std.fmt.allocPrint(gpa, "{s} by {s}, {s}, licensed under CC BY 4.0", .{ meta.title, author, repo });
    const doc = try std.fmt.allocPrint(gpa, "{s}. {s}\n\n{s}\n\nSource: {s}\nLicence: CC BY 4.0 ({s}). Credit: {s}.", .{
        meta.title, meta.summary, meta.io, repo, license_url, credit,
    });
    try pb.putBytes(&out, gpa, 2, producer);
    try pb.putBytes(&out, gpa, 3, version);
    try pb.putInt(&out, gpa, 5, meta.version);
    try pb.putBytes(&out, gpa, 6, doc);
    const values = [_][]const u8{ meta.title, author, repo, "CC-BY-4.0", license_url, credit, meta.training_data, meta.produced_by };
    for (keys, values) |key, value| {
        var prop: std.ArrayList(u8) = .empty;
        defer prop.deinit(gpa);
        try pb.putBytes(&prop, gpa, 1, key);
        try pb.putBytes(&prop, gpa, 2, value);
        try pb.putBytes(&out, gpa, 14, prop.items);
    }
    if (!std.mem.eql(u8, graph, try (try pb.field(out.items, 7)).?.bytes())) return error.GraphChanged;
    return out.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 3 or std.meta.stringToEnum(Model, args[1]) == null) {
        std.debug.print("usage: stamp_model_metadata scrfd|eye_refiner <model.onnx>\n", .{});
        return error.Usage;
    }
    const model = std.meta.stringToEnum(Model, args[1]).?;
    const stamped = try stamp(gpa, try u.readFile(gpa, init.io, args[2]), model);
    // Only replace the original after the complete stamped file has been written and flushed.
    const permissions = (try std.Io.Dir.cwd().statFile(init.io, args[2], .{})).permissions;
    var file = try std.Io.Dir.cwd().createFileAtomic(init.io, args[2], .{ .replace = true, .permissions = permissions });
    defer file.deinit(init.io);
    var buffer: [4096]u8 = undefined;
    var writer = file.file.writer(init.io, &buffer);
    try writer.interface.writeAll(stamped);
    try writer.flush();
    try file.replace(init.io);
    std.debug.print("stamped {s}: {s}, model_version {d}\n", .{ args[2], producer, info(model).version });
    var r: pb.Reader = .{ .data = stamped };
    while (try r.next()) |f| if (f.tag == 14) {
        std.debug.print("  {s}: {s}\n", .{ try pb.string(try f.bytes(), 1), try pb.string(try f.bytes(), 2) });
    };
}

test "stamping preserves graph, unknown fields, custom metadata and exporter; repeat is idempotent" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var model: std.ArrayList(u8) = .empty;
    try pb.putInt(&model, gpa, 1, 7);
    try pb.putBytes(&model, gpa, 2, "pytorch");
    try pb.putBytes(&model, gpa, 3, "2.0");
    try pb.putBytes(&model, gpa, 7, &.{ 18, 5, 'g', 'r', 'a', 'p', 'h' });
    try pb.putBytes(&model, gpa, 8, &.{ 16, 11 });
    try pb.putInt(&model, gpa, 999, 123);
    var custom: std.ArrayList(u8) = .empty;
    try pb.putBytes(&custom, gpa, 1, "custom");
    try pb.putBytes(&custom, gpa, 2, "retained");
    try pb.putBytes(&model, gpa, 14, custom.items);
    const first = try stamp(gpa, model.items, .scrfd);
    try std.testing.expectEqualStrings(try pb.string(model.items, 7), try pb.string(first, 7));
    try std.testing.expectEqualStrings("exported with pytorch 2.0", try pb.string(first, 3));
    try std.testing.expectEqualSlices(u8, (try pb.field(model.items, 999)).?.raw, (try pb.field(first, 999)).?.raw);
    var props: pb.Reader = .{ .data = first };
    var found_custom = false;
    while (try props.next()) |f| if (f.tag == 14 and std.mem.eql(u8, try pb.string(try f.bytes(), 1), "custom")) {
        try std.testing.expectEqualSlices(u8, custom.items, try f.bytes());
        found_custom = true;
    };
    try std.testing.expect(found_custom);
    try std.testing.expectEqualSlices(u8, first, try stamp(gpa, first, .scrfd));
    try std.testing.expectEqual(@as(u64, 1), try (try pb.field(try stamp(gpa, model.items, .eye_refiner), 5)).?.int());
}

test "malformed input and models without graphs fail before writing" {
    try std.testing.expectError(error.InvalidProtobuf, stamp(std.testing.allocator, &.{ 58, 255 }, .scrfd));
    try std.testing.expectError(error.MissingGraph, stamp(std.testing.allocator, &.{ 8, 7 }, .scrfd));
}
