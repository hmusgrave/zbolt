const std = @import("std");
const random = @import("./random.zig");
const Allocator = std.mem.Allocator;

fn centroid(comptime V: type, data: []V) V {
    var rtn = data[0];
    for (data[1..]) |x|
        rtn += x;
    const ti = @typeInfo(V).Vector;
    return rtn / @splat(ti.len, @intToFloat(ti.child, data.len));
}

fn all_centroids(comptime V: type, data: []V, assignment: []usize, comptime nc: usize) [nc]V {
    var rtn: [nc]V = undefined;
    var count = [_]usize{0} ** nc;
    for (data) |x, i| {
        const j = assignment[i];
        if (count[j] == 0) {
            rtn[j] = x;
        } else {
            rtn[j] += x;
        }
        count[j] += 1;
    }
    const ti = @typeInfo(V).Vector;
    for (rtn) |*x, i|
        x.* /= @splat(ti.len, @intToFloat(ti.child, count[i]));
    return rtn;
}

test "centroid" {
    const V = @Vector(4, f32);
    var data = [_]V{ V{ 1, 4, 7, 10 }, V{ -1, -2, -3, -4 } };
    var rtn = centroid(V, data[0..]);
    var i: usize = 0;
    while (i < 4) : (i += 1)
        try std.testing.expectEqual(rtn[i], @intToFloat(f32, i));
}

fn dot(comptime V: type, a: V, b: V) @typeInfo(V).Vector.child {
    return @reduce(.Add, a * b);
}

fn l2(comptime V: type, a: V, b: V) @typeInfo(V).Vector.child {
    return dot(V, a - b, a - b);
}

fn assign(comptime V: type, comptime ncenters: usize, centroids: [ncenters]V, data: []V, assignment: []usize) void {
    const ti = @typeInfo(V).Vector;
    _ = ncenters;
    for (assignment) |*x, i| {
        var best_i: usize = 0;
        var best_d: ti.child = l2(V, data[i], centroids[0]);
        for (centroids[1..]) |c, j| {
            const new_d: ti.child = l2(V, data[i], c);
            if (new_d < best_d) {
                best_i = j + 1;
                best_d = new_d;
            }
        }
        x.* = best_i;
    }
}

fn uniform_centers(allocator: Allocator, comptime V: type, _key: anytype, data: []V, comptime ncenters: usize) [ncenters]V {
    var key = _key;
    var choices = key.randint(usize, 0, data.len - 1, ncenters);
    var seen = std.hash_map.AutoHashMap(usize, void).init(allocator);
    defer seen.deinit();
    for (choices) |*c| {
        var attempt = c.*;
        while (seen.get(attempt) != null) {
            attempt = key.randint(usize, 0, data.len - 1, 1)[0];
        }
        seen.putNoClobber(attempt, {}) catch unreachable;
        c.* = attempt;
    }
    var centroids: [ncenters]V = undefined;
    for (centroids) |*x, i|
        x.* = data[choices[i]];
    return centroids;
}

fn square_centers(allocator: Allocator, comptime V: type, _key: anytype, data: []V, comptime ncenters: usize) ![ncenters]V {
    var key = _key;
    var keys = key.split(ncenters);
    var centroids: [ncenters]V = undefined;
    var dists: [ncenters][]f64 = blk: {
        var rtn: [ncenters][]f64 = undefined;
        for (rtn) |*v, i| {
            v.* = allocator.alloc(f64, data.len) catch |e| {
                for (rtn[0..i]) |*w|
                    allocator.free(w.*);
                return e;
            };
        }
        break :blk rtn;
    };
    defer {
        for (dists) |v|
            allocator.free(v);
    }
    var min_buf = try allocator.alloc(f64, data.len);
    defer allocator.free(min_buf);
    var seen = std.hash_map.AutoHashMap(usize, void).init(allocator);
    defer seen.deinit();
    for (centroids) |*c, i| {
        if (i == 0) {
            const idx = keys[i].randint(usize, 0, data.len - 1, 1)[0];
            c.* = data[idx];
            seen.putNoClobber(idx, {}) catch unreachable;
        } else {
            for (min_buf) |*m, j| {
                var least = dists[0][j];
                for (dists[1..i]) |d|
                    least = if (d[j] < least) d[j] else least;
                m.* = least;
            }
            var idx = keys[i].weighted_choice(f64, min_buf, 1)[0];
            while (seen.get(idx) != null) {
                idx = keys[i].weighted_choice(f64, min_buf, 1)[0];
            }
            c.* = data[idx];
            seen.putNoClobber(idx, {}) catch unreachable;
        }
        for (dists[i]) |*v, j|
            v.* = l2(V, c.*, data[j]);
    }
    return centroids;
}

fn _loyd(
    comptime V: type,
    data: []V,
    comptime ncenters: usize,
    _centroids: [ncenters]V,
    _assignment: []usize,
    _tmp_assignment: []usize,
) [ncenters]V {
    var centroids = _centroids;
    var assignment = _assignment;
    var tmp_assignment = _tmp_assignment;
    assign(V, ncenters, centroids, data, assignment);
    while (true) {
        centroids = all_centroids(V, data, assignment, ncenters);
        assign(V, ncenters, centroids, data, tmp_assignment);
        if (std.mem.eql(usize, assignment, tmp_assignment)) {
            return centroids;
        }
        var tmp = tmp_assignment;
        tmp_assignment = assignment;
        assignment = tmp;
    }
}

fn loyd_uniform(allocator: Allocator, comptime V: type, key: anytype, data: []V, comptime ncenters: usize) ![ncenters]V {
    var centroids = uniform_centers(allocator, V, key, data, ncenters);
    var assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(assignment);
    var tmp_assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(tmp_assignment);
    return _loyd(V, data, ncenters, centroids, assignment, tmp_assignment);
}

fn loyd_square(allocator: Allocator, comptime V: type, key: anytype, data: []V, comptime ncenters: usize) ![ncenters]V {
    var centroids = try square_centers(allocator, V, key, data, ncenters);
    var assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(assignment);
    var tmp_assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(tmp_assignment);
    return _loyd(V, data, ncenters, centroids, assignment, tmp_assignment);
}

test "loyd uniform should partition basic cluster" {
    var key = random.PRNGKey(random.Hashes.aes5){ .seed = 42 };
    const allocator = std.testing.allocator;
    const V = @Vector(4, f32);
    var data = [_]V{ V{ 0, 1, 2, 3 }, V{ 4, 5, 6, 7 }, V{ 0.5, 2.3, 2.4, 2.9 }, V{ 6, 5, 4, 8 } };
    var centers = try loyd_uniform(allocator, V, key, data[0..], 2);
    var assignment: [4]usize = undefined;
    assign(V, 2, centers, data[0..], assignment[0..]);
    try std.testing.expectEqual(assignment[0], assignment[2]);
    try std.testing.expectEqual(assignment[1], assignment[3]);
}

test "loyd square should partition basic cluster" {
    var key = random.PRNGKey(random.Hashes.aes5){ .seed = 42 };
    const allocator = std.testing.allocator;
    const V = @Vector(4, f32);
    var data = [_]V{ V{ 0, 1, 2, 3 }, V{ 4, 5, 6, 7 }, V{ 0.5, 2.3, 2.4, 2.9 }, V{ 6, 5, 4, 8 } };
    var centers = try loyd_square(allocator, V, key, data[0..], 2);
    var assignment: [4]usize = undefined;
    assign(V, 2, centers, data[0..], assignment[0..]);
    try std.testing.expectEqual(assignment[0], assignment[2]);
    try std.testing.expectEqual(assignment[1], assignment[3]);
}

// TODO: Handle duplicate data vectors
fn loyd_recursive(
    allocator: Allocator,
    comptime V: type,
    _key: anytype,
    data: []V,
    comptime ncenters: usize,
) Allocator.Error![ncenters]V {
    if (data.len <= ncenters * 8)
        return try loyd_square(allocator, V, _key, data[0..], ncenters);
    var key = _key;
    var keys = key.split(4);
    var partitions = try keys[0].randint_alloc(allocator, u2, 0, 1, data.len);
    defer allocator.free(partitions);
    var zcount: usize = 0;
    for (partitions) |x|
        zcount += @intCast(usize, x);
    zcount = data.len - zcount;
    var data1 = try allocator.alloc(V, zcount);
    defer allocator.free(data1);
    var data2 = try allocator.alloc(V, data.len - zcount);
    defer allocator.free(data2);
    var c1: usize = 0;
    var c2: usize = 0;
    for (partitions) |x, i| {
        if (x == 0) {
            data1[c1] = data[i];
            c1 += 1;
        } else {
            data2[c2] = data[i];
            c2 += 1;
        }
    }
    var centroids1 = try loyd_recursive(allocator, V, keys[1], data1, ncenters);
    var centroids2 = try loyd_recursive(allocator, V, keys[2], data2, ncenters);
    var max_centroids = try allocator.alloc(V, ncenters * 2);
    defer allocator.free(max_centroids);
    std.mem.copy(V, max_centroids, centroids1[0..]);
    std.mem.copy(V, max_centroids[ncenters..], centroids2[0..]);
    var centroids = try loyd_uniform(allocator, V, keys[3], max_centroids[0..], ncenters);
    var ta = try allocator.alloc(usize, data.len);
    defer allocator.free(ta);
    var tb = try allocator.alloc(usize, data.len);
    defer allocator.free(tb);
    return _loyd(V, data, ncenters, centroids, ta, tb);
}

test "loyd recursive" {
    var key = random.PRNGKey(random.Hashes.aes5){ .seed = 42 };
    const V = @Vector(4, f32);
    var data: [128]V = undefined;
    for (data) |*x, i| {
        const c = @intToFloat(f32, i);
        if (i < data.len >> 1) {
            x.* = V{ c, c + 1, c + 2, c + 3 };
        } else {
            x.* = V{ -c, -c + 1, -c + 2, -c + 3 };
        }
    }
    const allocator = std.testing.allocator;
    var centers = try loyd_recursive(allocator, V, key, data[0..], 2);
    var expected = [_]V{ V{ 31.5, 32.5, 33.5, 34.5 }, V{ -95.5, -94.5, -93.5, -92.5 } };
    try std.testing.expectEqual(l2(V, centers[0], expected[0]), 0.0);
    try std.testing.expectEqual(l2(V, centers[1], expected[1]), 0.0);
}
