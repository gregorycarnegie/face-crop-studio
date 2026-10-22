//! Helpers shared by the dataset tools: JSON numbers, whole-file reads, a CSV reader.
const std = @import("std");
const Io = std.Io;
const json = std.json;

pub fn readFile(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    return Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
}

pub fn writeFile(io: Io, path: []const u8, data: []const u8) !void {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

pub fn readJson(gpa: std.mem.Allocator, io: Io, path: []const u8) !json.Value {
    return json.parseFromSliceLeaky(json.Value, gpa, try readFile(gpa, io, path), .{});
}

/// A JSON number as f64 (Python's int/float distinction does not matter to these tools).
pub fn num(v: json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch std.math.nan(f64),
        else => std.math.nan(f64),
    };
}

pub fn str(v: json.Value) []const u8 {
    return v.string;
}

pub fn field(v: json.Value, name: []const u8) json.Value {
    return v.object.get(name).?;
}

pub fn joinPath(gpa: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    return std.fs.path.join(gpa, parts);
}

pub const Stdout = struct {
    buf: [4096]u8 = undefined,
    writer: Io.File.Writer,

    pub fn init(self: *Stdout, io: Io) void {
        self.writer = Io.File.stdout().writer(io, &self.buf);
    }

    pub fn w(self: *Stdout) *Io.Writer {
        return &self.writer.interface;
    }
};

/// A parsed CSV file: `rows[0]` is the header. Quoted fields, `""` escapes and embedded newlines
/// follow Python's `csv` defaults; fully empty lines are skipped as `DictReader` does.
pub const Csv = struct {
    header: []const []const u8,
    rows: []const []const []const u8,

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Csv {
        var rows: std.ArrayList([]const []const u8) = .empty;
        var row: std.ArrayList([]const u8) = .empty;
        var quoted: std.ArrayList(u8) = .empty; // only used for fields containing quotes
        var start: usize = 0; // field start in text
        var has_quote = false;
        var in_quotes = false;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            if (in_quotes) {
                if (c == '"') {
                    if (i + 1 < text.len and text[i + 1] == '"') {
                        try quoted.append(gpa, '"');
                        i += 1;
                    } else in_quotes = false;
                } else try quoted.append(gpa, c);
                continue;
            }
            if (c == '"') {
                has_quote = true;
                in_quotes = true;
            } else if (c == ',' or c == 13 or c == 10) {
                try row.append(gpa, if (has_quote) try quoted.toOwnedSlice(gpa) else text[start..i]);
                has_quote = false;
                if (c != ',') {
                    if (c == 13 and i + 1 < text.len and text[i + 1] == 10) i += 1;
                    // A blank line parses as one empty field; DictReader skips it.
                    if (!(row.items.len == 1 and row.items[0].len == 0)) try rows.append(gpa, try row.toOwnedSlice(gpa)) else row.clearRetainingCapacity();
                }
                start = i + 1;
            } else if (has_quote) try quoted.append(gpa, c);
        }
        if (start < text.len or row.items.len > 0) {
            try row.append(gpa, if (has_quote) try quoted.toOwnedSlice(gpa) else text[start..]);
            try rows.append(gpa, try row.toOwnedSlice(gpa));
        }
        const all = try rows.toOwnedSlice(gpa);
        if (all.len == 0) return error.EmptyCsv;
        return .{ .header = all[0], .rows = all[1..] };
    }

    pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Csv {
        return parse(gpa, try readFile(gpa, io, path));
    }

    pub fn col(self: Csv, name: []const u8) usize {
        for (self.header, 0..) |h, i| if (std.mem.eql(u8, h, name)) return i;
        std.debug.panic("CSV has no column {s}", .{name});
    }
};

pub fn lessStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// CPython's `random.Random(seed)` for a small non-negative integer seed (MT19937 seeded through
/// `init_by_array`), so a port shuffles exactly as the Python it replaces did.
pub const PyRandom = struct {
    mt: [624]u32 = undefined,
    idx: usize = 624,

    pub fn init(seed: u32) PyRandom {
        var r: PyRandom = .{};
        r.mt[0] = 19650218;
        for (1..624) |i| r.mt[i] = 1812433253 *% (r.mt[i - 1] ^ (r.mt[i - 1] >> 30)) +% @as(u32, @intCast(i));
        const key = [_]u32{seed};
        var i: usize = 1;
        var j: usize = 0;
        var k: usize = 624;
        while (k > 0) : (k -= 1) {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            i += 1;
            j += 1;
            if (i >= 624) {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if (j >= key.len) j = 0;
        }
        k = 623;
        while (k > 0) : (k -= 1) {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)) *% 1566083941)) -% @as(u32, @intCast(i));
            i += 1;
            if (i >= 624) {
                r.mt[0] = r.mt[623];
                i = 1;
            }
        }
        r.mt[0] = 0x80000000;
        return r;
    }

    fn next32(r: *PyRandom) u32 {
        if (r.idx >= 624) {
            for (0..624) |i| {
                const y = (r.mt[i] & 0x80000000) | (r.mt[(i + 1) % 624] & 0x7fffffff);
                r.mt[i] = r.mt[(i + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            r.idx = 0;
        }
        var y = r.mt[r.idx];
        r.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// `random.randrange(n)` for n < 2^32.
    pub fn below(r: *PyRandom, n: u32) u32 {
        const k: u32 = 32 - @clz(n);
        while (true) {
            const v = r.next32() >> @as(u5, @intCast(32 - k));
            if (v < n) return v;
        }
    }

    /// `random.sample(items, k)`, including CPython's switch between pool and selection-set.
    pub fn sample(r: *PyRandom, gpa: std.mem.Allocator, comptime T: type, items: []const T, k: usize) ![]T {
        const n = items.len;
        const result = try gpa.alloc(T, k);
        var setsize: usize = 21;
        if (k > 5) setsize += std.math.pow(usize, 4, @intFromFloat(@ceil(@log(@as(f64, @floatFromInt(k * 3))) / @log(4.0))));
        if (n <= setsize) {
            const pool = try gpa.dupe(T, items);
            for (0..k) |i| {
                const j = r.below(@intCast(n - i));
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            var selected: std.AutoHashMapUnmanaged(u32, void) = .empty;
            for (0..k) |i| {
                var j = r.below(@intCast(n));
                while (selected.contains(j)) j = r.below(@intCast(n));
                try selected.put(gpa, j, {});
                result[i] = items[j];
            }
        }
        return result;
    }

    pub fn shuffle(r: *PyRandom, comptime T: type, items: []T) void {
        var i = items.len;
        while (i > 1) {
            i -= 1;
            std.mem.swap(T, &items[i], &items[r.below(@intCast(i + 1))]);
        }
    }
};

/// Width and height from the first SOF marker, or null if `data` is not a JPEG.
pub fn jpegSize(data: []const u8) ?[2]u32 {
    if (data.len < 2 or data[0] != 0xFF or data[1] != 0xD8) return null;
    var i: usize = 2;
    while (true) {
        while (i < data.len and data[i] != 0xFF) i += 1;
        i += 1; // the 0xFF itself
        while (i < data.len and data[i] == 0xFF) i += 1;
        if (i >= data.len) return null;
        const m = data[i];
        i += 1;
        if (m >= 0xC0 and m <= 0xCF and m != 0xC4 and m != 0xC8 and m != 0xCC) {
            if (i + 7 > data.len) return null;
            return .{ std.mem.readInt(u16, data[i + 5 ..][0..2], .big), std.mem.readInt(u16, data[i + 3 ..][0..2], .big) };
        }
        if (i + 2 > data.len) return null;
        i += std.mem.readInt(u16, data[i..][0..2], .big);
    }
}

pub fn jpegSizeOfFile(gpa: std.mem.Allocator, io: Io, path: []const u8) ?[2]u32 {
    const data = readFile(gpa, io, path) catch return null;
    defer gpa.free(data);
    return jpegSize(data);
}

/// `v * 10^ndigits` rounded half to even, exactly: an f64 times a small power of ten fits an f128.
fn scaledRound(v: f64, comptime ndigits: u8) f128 {
    const s: f128 = @as(f128, v) * @as(f128, @floatFromInt(std.math.pow(u64, 10, ndigits)));
    const fl = @floor(s);
    const d = s - fl;
    if (d > 0.5) return fl + 1;
    if (d < 0.5) return fl;
    return if (@mod(fl, 2) == 0) fl else fl + 1;
}

/// Python's `round(v, ndigits)`: correctly rounded from the exact binary value, ties to even.
pub fn pyRound(v: f64, comptime ndigits: u8) f64 {
    const k = std.math.pow(f64, 10, ndigits);
    return @as(f64, @floatCast(scaledRound(v, ndigits))) / k;
}

/// Python's `f"{v:.{ndigits}f}"`; Zig's own `{d:.N}` rounds the shortest repr, which differs near ties.
pub fn fixed(w: *Io.Writer, v: f64, comptime ndigits: u8) !void {
    const r: u128 = @intFromFloat(scaledRound(@abs(v), ndigits));
    const k: u128 = std.math.pow(u128, 10, ndigits);
    if (std.math.signbit(v)) try w.writeAll("-");
    try w.print("{d}", .{r / k});
    if (ndigits > 0) try w.print(".{d:0>" ++ std.fmt.comptimePrint("{d}", .{ndigits}) ++ "}", .{r % k});
}

/// Write a float the way Python's `repr` / `json.dumps` does for ordinary magnitudes.
pub fn pyFloat(w: *Io.Writer, v: f64) !void {
    var buf: [64]u8 = undefined;
    const s = try std.fmt.bufPrint(&buf, "{d}", .{v});
    try w.writeAll(s);
    if (std.mem.indexOfAny(u8, s, ".eEni") == null) try w.writeAll(".0");
}

pub fn median(gpa: std.mem.Allocator, values: []const f64) !f64 {
    const s = try gpa.dupe(f64, values);
    std.mem.sort(f64, s, {}, std.sort.asc(f64));
    const n = s.len;
    return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2;
}

/// Records of a JSONL file keyed by `key`; a later line for the same key replaces the earlier
/// one but keeps its position, like a Python dict.
pub fn lastWins(gpa: std.mem.Allocator, io: Io, path: []const u8, key: []const u8) !std.StringArrayHashMapUnmanaged(json.Value) {
    var map: std.StringArrayHashMapUnmanaged(json.Value) = .empty;
    var it = std.mem.splitScalar(u8, try readFile(gpa, io, path), '\n');
    while (it.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len == 0) continue;
        const rec = try json.parseFromSliceLeaky(json.Value, gpa, line, .{});
        try map.put(gpa, rec.object.get(key).?.string, rec);
    }
    return map;
}

pub fn optField(v: json.Value, name: []const u8) ?json.Value {
    return v.object.get(name);
}

pub fn truthy(v: ?json.Value) bool {
    const x = v orelse return false;
    return switch (x) {
        .bool => |b| b,
        .null => false,
        .integer => |n| n != 0,
        else => true,
    };
}

/// `Path.as_uri()` for an absolute path.
pub fn asUri(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    var p = path;
    if (std.mem.startsWith(u8, p, "\\\\?\\")) p = p[4..];
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(gpa, if (p.len > 1 and p[1] == ':') "file:///" else "file://");
    for (p, 0..) |c, i| {
        const ch = if (c == '\\') '/' else c;
        if (std.ascii.isAlphanumeric(ch) or std.mem.indexOfScalar(u8, "-._~/", ch) != null or (ch == ':' and i == 1))
            try out.append(gpa, ch)
        else
            try out.print(gpa, "%{X:0>2}", .{ch});
    }
    return out.toOwnedSlice(gpa);
}

/// Contents of one member of a zip archive.
pub fn zipEntry(gpa: std.mem.Allocator, io: Io, zip_path: []const u8, name: []const u8) ![]u8 {
    var file = try Io.Dir.cwd().openFile(io, zip_path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);
    var it = try std.zip.Iterator.init(&reader);
    var name_buf: [std.fs.max_path_bytes]u8 = undefined;
    while (try it.next()) |entry| {
        if (!std.mem.eql(u8, try entry.getFilename(&reader, &name_buf, .{}), name)) continue;
        var out: Io.Writer.Allocating = .init(gpa);
        try entry.extractTo(&reader, &out.writer);
        return out.toOwnedSlice();
    }
    return error.FileNotFound;
}

/// `fixed` into a freshly allocated string, for padding or embedding in other output.
pub fn fixedAlloc(gpa: std.mem.Allocator, v: f64, comptime ndigits: u8) ![]u8 {
    var out: Io.Writer.Allocating = .init(gpa);
    try fixed(&out.writer, v, ndigits);
    return out.toOwnedSlice();
}

/// A string -> count map in first-seen order, like `collections.Counter`.
pub const Counter = struct {
    map: std.StringArrayHashMapUnmanaged(i64) = .empty,

    pub fn add(self: *Counter, gpa: std.mem.Allocator, key: []const u8, n: i64) !void {
        const e = try self.map.getOrPut(gpa, key);
        if (!e.found_existing) e.value_ptr.* = 0;
        e.value_ptr.* += n;
    }
    pub fn inc(self: *Counter, gpa: std.mem.Allocator, key: []const u8) !void {
        try self.add(gpa, key, 1);
    }
    pub fn get(self: Counter, key: []const u8) i64 {
        return self.map.get(key) orelse 0;
    }
};

pub fn padLeft(w: *Io.Writer, s: []const u8, width: usize) !void {
    if (s.len < width) try w.splatByteAll(' ', width - s.len);
    try w.writeAll(s);
}

pub fn padRight(w: *Io.Writer, s: []const u8, width: usize) !void {
    try w.writeAll(s);
    if (s.len < width) try w.splatByteAll(' ', width - s.len);
}
