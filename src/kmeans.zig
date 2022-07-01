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
    var data = [_]V{V{1, 4, 7, 10}, V{-1, -2, -3, -4}};
    var rtn = centroid(V, data[0..]);
    var i: usize = 0;
    while (i < 4) : (i += 1)
        try std.testing.expectEqual(rtn[i], @intToFloat(f32, i));
}

fn dot(comptime V: type, a: V, b: V) @typeInfo(V).Vector.child {
    return @reduce(.Add, a*b);
}

fn l2(comptime V: type, a: V, b: V) @typeInfo(V).Vector.child {
    return dot(V, a-b, a-b);
}

fn assign(comptime V: type, comptime ncenters: usize, centroids: [ncenters]V,
data: []V, assignment: []usize) void {
    const ti = @typeInfo(V).Vector;
    _ = ncenters;
    for (assignment) |*x, i| {
        var best_i: usize = 0;
        var best_d: ti.child = l2(V, data[i], centroids[0]);
        for (centroids[1..]) |c,j| {
            const new_d: ti.child = l2(V, data[i], c);
            if (new_d < best_d) {
                best_i = j+1;
                best_d = new_d;
            }
        }
        x.* = best_i;
    }
}

fn loyd(
    allocator: Allocator,
    comptime V: type,
    key: anytype,
    data: []V,
    comptime ncenters: usize
) ![ncenters]V {
    var foo = key;
    var choices = foo.randint(usize, 0, data.len, ncenters);
    var centroids: [ncenters]V = undefined;
    for (centroids) |*x, i|
        x.* = data[choices[i]];
    var assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(assignment);
    var tmp_assignment = try allocator.alloc(usize, data.len);
    defer allocator.free(tmp_assignment);
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

test "loyd" {
    var key = random.PRNGKey(random.Hashes.aes5){.seed=42};
    const allocator = std.testing.allocator;
    const V = @Vector(4, f32);
    var data = [_]V{V{0, 1, 2, 3}, V{4, 5, 6, 7}, V{0.5, 2.3, 2.4, 2.9}, V{6,
    5, 4, 8}};
    var foo = try loyd(allocator, V, key, data[0..], 2);
    std.debug.print("{any}\n", .{foo});
}
