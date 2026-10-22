//! Rank likely "god files" by size, complexity, and breadth metrics.
//!
//!   zig run -O ReleaseFast tools/find_god_files.zig -- [roots...] [--min-score N] [--top N]
//!       [--json] [--include-tests] [--extensions .rs,.py] [--skip-dir NAME]...
//!
//! The Python original used regexes; the matchers below hand-code the same patterns, so a
//! change to a pattern there is a change to the matching function here.
const std = @import("std");
const Io = std.Io;

const SOURCE_EXTENSIONS = [_][]const u8{ ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js", ".jsx", ".kt", ".py", ".rs", ".ts", ".tsx" };
const SKIP_DIRS = [_][]const u8{ ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".svn", ".venv", "__pycache__", "build", "coverage", "dist", "node_modules", "target", "vendor" };
const DEFAULT_MIN_SCORE = 100.0;

const Metrics = struct {
    total_lines: i64 = 0,
    code_lines: i64 = 0,
    functions: i64 = 0,
    types: i64 = 0,
    impl_blocks: i64 = 0,
    decision_points: i64 = 0,
    imports: i64 = 0,
    fanout: i64 = 0,
    max_function_lines: i64 = 0,
};

const Reason = struct { label: []const u8, limit: i64, field: []const u8 };
const REASONS = [_]Reason{
    .{ .label = "total lines", .limit = 500, .field = "total_lines" },
    .{ .label = "code lines", .limit = 350, .field = "code_lines" },
    .{ .label = "functions", .limit = 25, .field = "functions" },
    .{ .label = "types", .limit = 12, .field = "types" },
    .{ .label = "impl blocks", .limit = 15, .field = "impl_blocks" },
    .{ .label = "decision points", .limit = 90, .field = "decision_points" },
    .{ .label = "imports", .limit = 45, .field = "imports" },
    .{ .label = "import fanout", .limit = 25, .field = "fanout" },
    .{ .label = "longest function", .limit = 120, .field = "max_function_lines" },
};

const Lang = enum { py, js, other };

fn langOf(suffix: []const u8) Lang {
    if (std.mem.eql(u8, suffix, ".py")) return .py;
    for ([_][]const u8{ ".js", ".jsx", ".ts", ".tsx" }) |s| if (std.mem.eql(u8, suffix, s)) return .js;
    return .other;
}

// ---- tiny matcher kit: each helper returns the end index of a match at `i`, or null ----

fn isWs(c: u8) bool {
    return c == ' ' or (c >= 9 and c <= 13);
}
fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c >= 0x80;
}
fn isIdentStart(c: u8, dollar: bool) bool {
    return std.ascii.isAlphabetic(c) or c == '_' or (dollar and c == '$');
}
fn wsStar(t: []const u8, i: usize) usize {
    var j = i;
    while (j < t.len and isWs(t[j])) j += 1;
    return j;
}
fn wsPlus(t: []const u8, i: usize) ?usize {
    const j = wsStar(t, i);
    return if (j > i) j else null;
}
fn lit(t: []const u8, i: usize, s: []const u8) ?usize {
    return if (std.mem.startsWith(u8, t[i..], s)) i + s.len else null;
}
fn ident(t: []const u8, i: usize, dollar: bool) ?usize {
    if (i >= t.len or !isIdentStart(t[i], dollar)) return null;
    var j = i + 1;
    while (j < t.len and (isWord(t[j]) or (dollar and t[j] == '$'))) j += 1;
    return j;
}
fn wordBoundaryBefore(t: []const u8, i: usize) bool {
    return i == 0 or !isWord(t[i - 1]);
}
fn lineStart(t: []const u8, i: usize) bool {
    return i == 0 or t[i - 1] == '\n';
}

/// `kw\s+IDENT\s*(` style: keyword, whitespace, identifier, optional whitespace, then one of `after`.
fn kwIdentThen(t: []const u8, i: usize, kw: []const u8, dollar: bool, after: []const u8) ?usize {
    var j = lit(t, i, kw) orelse return null;
    j = wsPlus(t, j) orelse return null;
    j = ident(t, j, dollar) orelse return null;
    j = wsStar(t, j);
    if (j < t.len and std.mem.indexOfScalar(u8, after, t[j]) != null) return j + 1;
    return null;
}

/// Count non-overlapping matches of `matcher` over the whole text, trying every start position
/// (or only line starts when `anchored`), the way `re.findall` scans.
fn countMatches(t: []const u8, anchored: bool, wordB: bool, matcher: *const fn ([]const u8, usize) ?usize) i64 {
    var n: i64 = 0;
    var i: usize = 0;
    while (i < t.len) {
        if ((!anchored or lineStart(t, i)) and (!wordB or wordBoundaryBefore(t, i))) {
            if (matcher(t, i)) |end| {
                n += 1;
                i = @max(end, i + 1);
                continue;
            }
        }
        i += 1;
    }
    return n;
}

fn mPyDef(t: []const u8, i: usize) ?usize {
    var j = wsStar(t, i);
    if (lit(t, j, "async")) |k| if (wsPlus(t, k)) |k2| if (lit(t, k2, "def") != null) {
        j = k2;
    };
    return kwIdentThen(t, j, "def", false, "(");
}
fn mPyClass(t: []const u8, i: usize) ?usize {
    const j = lit(t, wsStar(t, i), "class") orelse return null;
    return ident(t, wsPlus(t, j) orelse return null, false);
}
fn mJsFunction(t: []const u8, i: usize) ?usize {
    return kwIdentThen(t, i, "function", true, "(");
}
fn mJsVar(t: []const u8, i: usize) ?usize {
    var j = wsStar(t, i);
    if (lit(t, j, "export")) |k| if (wsPlus(t, k)) |k2| {
        j = k2;
    };
    const kws = [_][]const u8{ "const", "let", "var" };
    for (kws) |kw| {
        if (lit(t, j, kw)) |k| {
            const k2 = wsPlus(t, k) orelse continue;
            const k3 = ident(t, k2, true) orelse continue;
            const k4 = wsStar(t, k3);
            if (k4 < t.len and t[k4] == '=') return k4 + 1;
        }
    }
    // the optional `export` may have been wrongly consumed; a regex would backtrack
    if (j != wsStar(t, i)) return mJsVarNoExport(t, i);
    return null;
}
fn mJsVarNoExport(t: []const u8, i: usize) ?usize {
    const j = wsStar(t, i);
    for ([_][]const u8{ "const", "let", "var" }) |kw| {
        if (lit(t, j, kw)) |k| {
            const k2 = wsPlus(t, k) orelse continue;
            const k3 = ident(t, k2, true) orelse continue;
            const k4 = wsStar(t, k3);
            if (k4 < t.len and t[k4] == '=') return k4 + 1;
        }
    }
    return null;
}
fn mFn(t: []const u8, i: usize) ?usize {
    return kwIdentThen(t, i, "fn", false, "<(");
}
fn mJsType(t: []const u8, i: usize) ?usize {
    for ([_][]const u8{ "class", "interface", "type" }) |kw| {
        if (lit(t, i, kw)) |k| if (wsPlus(t, k)) |k2| if (ident(t, k2, true)) |e| return e;
    }
    return null;
}
fn mOtherType(t: []const u8, i: usize) ?usize {
    for ([_][]const u8{ "struct", "enum", "trait", "type" }) |kw| {
        if (lit(t, i, kw)) |k| if (wsPlus(t, k)) |k2| if (ident(t, k2, false)) |e| return e;
    }
    return null;
}
fn mPyImport(t: []const u8, i: usize) ?usize {
    const j = wsStar(t, i);
    if (lit(t, j, "from")) |k| if (wsPlus(t, k)) |k2| {
        var m = k2;
        while (m < t.len and !isWs(t[m])) m += 1; // \S+
        if (m > k2) if (wsPlus(t, m)) |m2| if (lit(t, m2, "import")) |m3| if (wsPlus(t, m3)) |e| return e;
        // backtracking \S+ shorter cannot help: it would have to end before a whitespace
    };
    if (lit(t, j, "import")) |k| return wsPlus(t, k);
    return null;
}
fn mKeywordLine(comptime kw: []const u8) *const fn ([]const u8, usize) ?usize {
    return struct {
        fn m(t: []const u8, i: usize) ?usize {
            const j = lit(t, wsStar(t, i), kw) orelse return null;
            return wsPlus(t, j);
        }
    }.m;
}
fn mImpl(t: []const u8, i: usize) ?usize {
    const j = lit(t, i, "impl") orelse return null;
    // optional generics: \s*<[^>{;]*>
    const k = wsStar(t, j);
    if (k < t.len and t[k] == '<') {
        var m = k + 1;
        while (m < t.len and t[m] != '>' and t[m] != '{' and t[m] != ';') m += 1;
        if (m < t.len and t[m] == '>') if (wsPlus(t, m + 1)) |e| return e;
    }
    return wsPlus(t, j);
}

fn countDecisions(t: []const u8) i64 {
    var n: i64 = 0;
    var i: usize = 0;
    const kws = [_][]const u8{ "if", "match", "for", "while", "loop", "switch", "case", "catch" };
    while (i < t.len) : (i += 1) {
        if (!wordBoundaryBefore(t, i) or !std.ascii.isAlphabetic(t[i])) continue;
        // alternation order: if | else\s+if | match | for | ...
        var end: ?usize = null;
        if (lit(t, i, "if")) |e| {
            end = e;
        } else if (lit(t, i, "else")) |e| {
            if (wsPlus(t, e)) |e2| end = lit(t, e2, "if");
        }
        if (end == null) for (kws[1..]) |kw| {
            if (lit(t, i, kw)) |e| {
                end = e;
                break;
            }
        };
        const e = end orelse continue;
        if (e < t.len and isWord(t[e])) continue;
        n += 1;
        i = e - 1;
    }
    n += @intCast(std.mem.count(u8, t, "&&"));
    n += @intCast(std.mem.count(u8, t, "||"));
    n += @intCast(std.mem.count(u8, t, "?"));
    return n;
}

/// `str.splitlines()` on the characters that matter for source files.
fn splitLines(gpa: std.mem.Allocator, t: []const u8) ![][]const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < t.len) {
        const c = t[i];
        var brk: usize = 0; // bytes consumed by the separator
        if (c == '\r') {
            brk = if (i + 1 < t.len and t[i + 1] == '\n') 2 else 1;
        } else if (c == '\n' or c == 0x0b or c == 0x0c or (c >= 0x1c and c <= 0x1e)) {
            brk = 1;
        } else if (c == 0xC2 and i + 1 < t.len and t[i + 1] == 0x85) {
            brk = 2;
        } else if (c == 0xE2 and i + 2 < t.len and t[i + 1] == 0x80 and (t[i + 2] == 0xA8 or t[i + 2] == 0xA9)) {
            brk = 3;
        }
        if (brk > 0) {
            try lines.append(gpa, t[start..i]);
            i += brk;
            start = i;
        } else i += 1;
    }
    if (start < t.len) try lines.append(gpa, t[start..]);
    return lines.toOwnedSlice(gpa);
}

fn stripComments(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    // /* ... */ -> as many newlines as it held
    var nb: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "/*")) {
            if (std.mem.indexOfPos(u8, text, i + 2, "*/")) |end| {
                for (text[i .. end + 2]) |c| if (c == '\n') try nb.append(gpa, '\n');
                i = end + 2;
                continue;
            }
        }
        try nb.append(gpa, text[i]);
        i += 1;
    }
    var out: std.ArrayList(u8) = .empty;
    const lines = try splitLines(gpa, nb.items);
    for (lines, 0..) |line, n| {
        var l = line;
        if (std.mem.indexOf(u8, l, "//")) |k| l = l[0..k];
        if (std.mem.indexOfScalar(u8, l, '#')) |k| l = l[0..k];
        if (n > 0) try out.append(gpa, '\n');
        try out.appendSlice(gpa, l);
    }
    return out.toOwnedSlice(gpa);
}

fn looksLikeFunctionStart(line: []const u8, lang: Lang) bool {
    switch (lang) {
        .py => return mPyDef(line, 0) != null,
        .js => {
            for (0..line.len) |i| if (wordBoundaryBefore(line, i) and mJsFunction(line, i) != null) return true;
            return false;
        },
        .other => {
            for (0..line.len) |i| if (wordBoundaryBefore(line, i) and mFn(line, i) != null) return true;
            return false;
        },
    }
}

fn maxFunctionLines(gpa: std.mem.Allocator, lines: []const []const u8, lang: Lang) !i64 {
    var starts: std.ArrayList(usize) = .empty;
    for (lines, 0..) |l, i| if (looksLikeFunctionStart(l, lang)) try starts.append(gpa, i);
    var best: i64 = 0;
    for (starts.items) |start| {
        var depth: i64 = 0;
        var saw_body = false;
        var end = start;
        while (end < lines.len) : (end += 1) {
            if (std.mem.indexOfScalar(u8, lines[end], '{') != null) saw_body = true;
            depth += @as(i64, @intCast(std.mem.count(u8, lines[end], "{"))) - @as(i64, @intCast(std.mem.count(u8, lines[end], "}")));
            if (saw_body and depth <= 0) {
                best = @max(best, @as(i64, @intCast(end - start + 1)));
                break;
            }
        }
    }
    if (lang == .py) {
        for (starts.items, 0..) |start, idx| {
            const end = if (idx + 1 < starts.items.len) starts.items[idx + 1] else lines.len;
            best = @max(best, @as(i64, @intCast(end - start)));
        }
    }
    return best;
}

fn trimSet(s: []const u8, set: []const u8) []const u8 {
    return std.mem.trim(u8, s, set);
}

fn countFanout(gpa: std.mem.Allocator, t: []const u8, lang: Lang) !i64 {
    var modules: std.StringHashMapUnmanaged(void) = .empty;
    var i: usize = 0;
    switch (lang) {
        .py => while (i < t.len) : (i += 1) {
            if (!lineStart(t, i)) continue;
            const j = wsStar(t, i);
            var name: ?[]const u8 = null;
            if (lit(t, j, "from")) |k| if (wsPlus(t, k)) |k2| {
                var m = k2;
                while (m < t.len and (std.ascii.isAlphanumeric(t[m]) or t[m] == '_' or t[m] == '.')) m += 1;
                if (m > k2) if (wsPlus(t, m)) |m2| if (lit(t, m2, "import") != null) {
                    name = t[k2..m];
                };
            };
            if (name == null) if (lit(t, j, "import")) |k| if (wsPlus(t, k)) |k2| {
                var m = k2;
                while (m < t.len and (std.ascii.isAlphanumeric(t[m]) or t[m] == '_' or t[m] == '.')) m += 1;
                if (m > k2) name = t[k2..m];
            };
            if (name) |n| try modules.put(gpa, n[0 .. std.mem.indexOfScalar(u8, n, '.') orelse n.len], {});
        },
        .js => while (i < t.len) : (i += 1) {
            if (!lineStart(t, i)) continue;
            const j = lit(t, wsStar(t, i), "import") orelse continue;
            const k = wsPlus(t, j) orelse continue;
            const eol = std.mem.indexOfScalarPos(u8, t, k, '\n') orelse t.len;
            const line = t[k..eol];
            // lazy `.*?\s+from\s+['"]([^'"]+)['"]` -- the first `from` that fits wins
            var from: usize = 0;
            while (std.mem.indexOfPos(u8, line, from, "from")) |f| : (from = f + 1) {
                if (f == 0 or !isWs(line[f - 1])) continue;
                const a = wsPlus(line, f + 4) orelse continue;
                if (a >= line.len or (line[a] != '\'' and line[a] != '"')) continue;
                const q = std.mem.indexOfAnyPos(u8, line, a + 1, "'\"") orelse continue;
                if (q > a + 1) {
                    try modules.put(gpa, line[a + 1 .. q], {});
                    break;
                }
            }
        },
        .other => while (i < t.len) : (i += 1) {
            if (!lineStart(t, i)) continue;
            const j = lit(t, wsStar(t, i), "use") orelse continue;
            const k = wsPlus(t, j) orelse continue;
            const semi = std.mem.indexOfScalarPos(u8, t, k, ';') orelse continue;
            if (semi == k) continue;
            var path = trimSet(t[k..semi], " \t\n\r\x0b\x0c");
            path = path[0 .. std.mem.indexOfScalar(u8, path, '{') orelse path.len];
            path = trimSet(path, ": ");
            var parts = std.mem.splitSequence(u8, path, "::");
            const first = parts.next().?;
            const second = parts.next();
            if ((std.mem.eql(u8, first, "crate") or std.mem.eql(u8, first, "self") or std.mem.eql(u8, first, "super")) and second != null) {
                try modules.put(gpa, try std.fmt.allocPrint(gpa, "{s}::{s}", .{ first, second.? }), {});
            } else try modules.put(gpa, first, {});
        },
    }
    return modules.count();
}

fn analyze(gpa: std.mem.Allocator, text: []const u8, suffix: []const u8) !Metrics {
    const lang = langOf(suffix);
    const code = try stripComments(gpa, text);
    const code_lines_list = try splitLines(gpa, code);
    var m: Metrics = .{};
    m.total_lines = @intCast((try splitLines(gpa, text)).len);
    for (code_lines_list) |l| m.code_lines += @intFromBool(std.mem.trim(u8, l, " \t\r\x0b\x0c").len > 0);
    switch (lang) {
        .py => {
            m.functions = countMatches(code, true, false, mPyDef);
            m.types = countMatches(code, true, false, mPyClass);
            m.imports = countMatches(code, true, false, mPyImport);
        },
        .js => {
            m.functions = countMatches(code, false, true, mJsFunction) + countMatches(code, true, false, mJsVar);
            m.types = countMatches(code, false, true, mJsType);
            m.imports = countMatches(code, true, false, mKeywordLine("import"));
        },
        .other => {
            m.functions = countMatches(code, false, true, mFn);
            m.types = countMatches(code, false, true, mOtherType);
            m.imports = countMatches(code, true, false, mKeywordLine("use"));
        },
    }
    if (std.mem.eql(u8, suffix, ".rs")) m.impl_blocks = countMatches(code, false, true, mImpl);
    m.decision_points = countDecisions(code);
    m.fanout = try countFanout(gpa, code, lang);
    m.max_function_lines = try maxFunctionLines(gpa, code_lines_list, lang);
    return m;
}

fn over(v: i64, floor: i64, weight: f64) f64 {
    return @as(f64, @floatFromInt(@max(0, v - floor))) * weight;
}

fn score(m: Metrics) f64 {
    var v: f64 = 0.0;
    v += over(m.total_lines, 300, 0.20);
    v += over(m.code_lines, 220, 0.30);
    v += over(m.functions, 15, 4.00);
    v += over(m.types, 6, 6.00);
    v += over(m.impl_blocks, 8, 3.00);
    v += over(m.decision_points, 50, 1.20);
    v += over(m.imports, 25, 1.50);
    v += over(m.fanout, 15, 2.50);
    v += over(m.max_function_lines, 80, 0.80);
    return @floor(v * 10 + 0.5) / 10; // round(v, 1); weights are short decimals, so ties are not a concern
}

const Finding = struct { path: []const u8, score: f64, m: Metrics, reasons: [][]const u8 };

fn lessFinding(_: void, a: Finding, b: Finding) bool {
    if (a.score != b.score) return a.score > b.score;
    if (a.m.total_lines != b.m.total_lines) return a.m.total_lines > b.m.total_lines;
    return std.mem.lessThan(u8, a.path, b.path);
}

fn suffixOf(name: []const u8) []const u8 {
    const dot = std.mem.findScalarLast(u8, name, '.') orelse return "";
    return if (dot == 0) "" else name[dot..];
}

fn looksTestOnly(path: []const u8, name: []const u8, gpa: std.mem.Allocator) !bool {
    var it = std.mem.tokenizeAny(u8, path, "/\\");
    while (it.next()) |part| {
        const lower = try std.ascii.allocLowerString(gpa, part);
        if (std.mem.eql(u8, lower, "tests") or std.mem.eql(u8, lower, "__tests__")) return true;
    }
    const n = try std.ascii.allocLowerString(gpa, name);
    if (std.mem.eql(u8, n, "tests.rs") or std.mem.startsWith(u8, n, "test_")) return true;
    for ([_][]const u8{ "_test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx" }) |s| if (std.mem.endsWith(u8, n, s)) return true;
    return false;
}

fn inSkipDirs(path: []const u8, skip: []const []const u8) bool {
    var it = std.mem.tokenizeAny(u8, path, "/\\");
    while (it.next()) |part| for (skip) |s| if (std.mem.eql(u8, part, s)) return true;
    return false;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);

    var roots: std.ArrayList([]const u8) = .empty;
    var min_score: f64 = DEFAULT_MIN_SCORE;
    var top: usize = 0;
    var as_json = false;
    var include_tests = false;
    var ext_arg: ?[]const u8 = null;
    var skip: std.ArrayList([]const u8) = .empty;
    try skip.appendSlice(gpa, &SKIP_DIRS);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--min-score") and i + 1 < args.len) {
            i += 1;
            min_score = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, a, "--top") and i + 1 < args.len) {
            i += 1;
            top = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, a, "--json")) {
            as_json = true;
        } else if (std.mem.eql(u8, a, "--include-tests")) {
            include_tests = true;
        } else if (std.mem.eql(u8, a, "--extensions") and i + 1 < args.len) {
            i += 1;
            ext_arg = args[i];
        } else if (std.mem.eql(u8, a, "--skip-dir") and i + 1 < args.len) {
            i += 1;
            try skip.append(gpa, args[i]);
        } else try roots.append(gpa, a);
    }
    if (roots.items.len == 0) try roots.append(gpa, ".");

    var exts: std.ArrayList([]const u8) = .empty;
    if (ext_arg) |e| {
        var it = std.mem.splitScalar(u8, e, ',');
        while (it.next()) |p| {
            const t = std.mem.trim(u8, p, " \t");
            if (t.len == 0) continue;
            try exts.append(gpa, if (t[0] == '.') t else try std.fmt.allocPrint(gpa, ".{s}", .{t}));
        }
    } else try exts.appendSlice(gpa, &SOURCE_EXTENSIONS);

    const cwd = Io.Dir.cwd();
    const repo_root = try cwd.realPathFileAlloc(io, ".", gpa);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var findings: std.ArrayList(Finding) = .empty;

    for (roots.items) |root| {
        var candidates: std.ArrayList([]const u8) = .empty;
        const st = cwd.statFile(io, root, .{}) catch |e| {
            std.debug.print("cannot read {s}: {s}\n", .{ root, @errorName(e) });
            continue;
        };
        if (st.kind == .directory) {
            var dir = try cwd.openDir(io, root, .{ .iterate = true });
            defer dir.close(io);
            var walker = try dir.walk(gpa);
            while (try walker.next(io)) |e| {
                if (e.kind != .file) continue;
                try candidates.append(gpa, try std.fs.path.join(gpa, &.{ root, e.path }));
            }
        } else try candidates.append(gpa, root);

        for (candidates.items) |path| {
            const name = std.fs.path.basename(path);
            const suffix = suffixOf(name);
            const wanted = for (exts.items) |e| {
                if (std.mem.eql(u8, e, suffix)) break true;
            } else false;
            if (!wanted) continue;
            const resolved = cwd.realPathFileAlloc(io, path, gpa) catch continue;
            if (seen.contains(resolved)) continue;
            if (inSkipDirs(path, skip.items)) continue;
            if (!include_tests and try looksTestOnly(path, name, gpa)) continue;
            try seen.put(gpa, resolved, {});

            const raw = cwd.readFileAlloc(io, path, gpa, .unlimited) catch continue;
            const text = try validUtf8(gpa, raw);
            const m = try analyze(gpa, text, suffix);
            const s = score(m);
            if (s < min_score) continue;

            var reasons: std.ArrayList([]const u8) = .empty;
            inline for (REASONS) |r| {
                if (@field(m, r.field) >= r.limit)
                    try reasons.append(gpa, try std.fmt.allocPrint(gpa, "{s} {d} >= {d}", .{ r.label, @field(m, r.field), r.limit }));
            }
            try reasons.append(gpa, try std.fmt.allocPrint(gpa, "score {s} >= {s}", .{ try fixed1(gpa, s), try fixed1(gpa, min_score) }));

            var rel: []const u8 = resolved;
            if (std.mem.startsWith(u8, rel, repo_root)) rel = std.mem.trimStart(u8, rel[repo_root.len..], "/\\");
            const posix = try gpa.dupe(u8, rel);
            std.mem.replaceScalar(u8, posix, '\\', '/');
            try findings.append(gpa, .{ .path = posix, .score = s, .m = m, .reasons = try reasons.toOwnedSlice(gpa) });
        }
    }

    std.mem.sort(Finding, findings.items, {}, lessFinding);
    const shown = if (top > 0) findings.items[0..@min(top, findings.items.len)] else findings.items;

    var so_buf: [4096]u8 = undefined;
    var so = Io.File.stdout().writer(io, &so_buf);
    const w = &so.interface;
    if (as_json) try printJson(gpa, w, shown) else try printTable(gpa, w, shown);
    try w.flush();
}

fn fixed1(gpa: std.mem.Allocator, v: f64) ![]u8 {
    return std.fmt.allocPrint(gpa, "{d:.1}", .{v});
}

/// `errors="ignore"`: drop bytes that are not valid UTF-8.
fn validUtf8(gpa: std.mem.Allocator, raw: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(raw)) return raw;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < raw.len) {
        const n = std.unicode.utf8ByteSequenceLength(raw[i]) catch {
            i += 1;
            continue;
        };
        if (i + n <= raw.len and std.unicode.utf8ValidateSlice(raw[i .. i + n])) {
            try out.appendSlice(gpa, raw[i .. i + n]);
            i += n;
        } else i += 1;
    }
    return out.toOwnedSlice(gpa);
}

fn printTable(gpa: std.mem.Allocator, w: *Io.Writer, findings: []const Finding) !void {
    if (findings.len == 0) return w.writeAll("No god files found with the current thresholds.\n");
    const header = "score  lines  code  fn  type  impl  branch  use  fanout  max_fn  path";
    try w.print("{s}\n", .{header});
    try w.splatByteAll('-', header.len);
    try w.writeAll("\n");
    for (findings) |f| {
        const m = f.m;
        const score_s = try fixed1(gpa, f.score);
        try w.writeAll("");
        try pad(w, gpa, "{s}", .{score_s}, 5);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.total_lines}, 5);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.code_lines}, 4);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.functions}, 2);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.types}, 4);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.impl_blocks}, 4);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.decision_points}, 6);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.imports}, 3);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.fanout}, 6);
        try w.writeAll("  ");
        try pad(w, gpa, "{d}", .{m.max_function_lines}, 6);
        try w.print("  {s}\n       reasons: ", .{f.path});
        for (f.reasons, 0..) |r, i| try w.print("{s}{s}", .{ if (i > 0) "; " else "", r });
        try w.writeAll("\n");
    }
    try w.writeAll("\nFiles:\n");
    for (findings) |f| try w.print("{s}\n", .{f.path});
}

fn pad(w: *Io.Writer, gpa: std.mem.Allocator, comptime fmt: []const u8, args: anytype, width: usize) !void {
    const s = try std.fmt.allocPrint(gpa, fmt, args);
    if (s.len < width) try w.splatByteAll(' ', width - s.len);
    try w.writeAll(s);
}

fn printJson(gpa: std.mem.Allocator, w: *Io.Writer, findings: []const Finding) !void {
    _ = gpa;
    if (findings.len == 0) return w.writeAll("[]\n");
    try w.writeAll("[\n");
    for (findings, 0..) |f, n| {
        const m = f.m;
        try w.print("  {{\n    \"path\": \"{s}\",\n    \"score\": ", .{f.path});
        var buf: [64]u8 = undefined;
        const s = try std.fmt.bufPrint(&buf, "{d}", .{f.score});
        try w.writeAll(s);
        if (std.mem.indexOfScalar(u8, s, '.') == null) try w.writeAll(".0");
        try w.print(",\n    \"total_lines\": {d},\n    \"code_lines\": {d},\n    \"functions\": {d},\n    \"types\": {d},\n    \"impl_blocks\": {d},\n    \"decision_points\": {d},\n    \"imports\": {d},\n    \"fanout\": {d},\n    \"max_function_lines\": {d},\n    \"reasons\": [\n", .{ m.total_lines, m.code_lines, m.functions, m.types, m.impl_blocks, m.decision_points, m.imports, m.fanout, m.max_function_lines });
        for (f.reasons, 0..) |r, i| try w.print("      \"{s}\"{s}\n", .{ r, if (i + 1 < f.reasons.len) "," else "" });
        try w.print("    ]\n  }}{s}\n", .{if (n + 1 < findings.len) "," else ""});
    }
    try w.writeAll("]\n");
}
