const std = @import("std");
const Allocator = std.mem.Allocator;
const kmeans = @import("./kmeans.zig");
const random = @import("./random.zig");

pub fn Database(comptime V: type) type {
    return struct {
        centroids: [16]V,
        data: []u4,  // TODO: packed data
        allocator: Allocator,

        pub fn init(allocator: Allocator, key: anytype, data: []V) !@This() {
            var centers = try kmeans.loyd_recursive(allocator, V, key, data, 16);
            var assignment = try allocator.alloc(usize, data.len);
            defer allocator.free(assignment);
            kmeans.assign(V, 16, centers, data, assignment);
            var new_data = try allocator.alloc(u4, data.len);
            errdefer allocator.free(new_data);
            for (new_data) |*x, i|
                x.* = @intCast(u4, assignment[i]);
            return @This() {
                .centroids = centers,
                .data = new_data,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.data);
        }

        pub fn encode(self: *@This(), query: V) [16]f32 {
            var rtn: [16]f32 = undefined;
            for (rtn) |*r, i|
                r.* = @reduce(.Add, query * self.centroids[i]);
            return rtn;
        }

        pub fn apply(self: *@This(), allocator: Allocator, query: [16]f32) ![]f32 {
            var rtn = try allocator.alloc(f32, self.data.len);
            for (rtn) |*r, i|
                r.* = query[@intCast(usize, self.data[i])];
            return rtn;
        }
    };
}

test {
    const V = @Vector(100, f32);
    std.debug.print("\n", .{});
    var data: [200]V = undefined;
    for (data) |*x, i| {
        var v: [100]f32 = undefined;
        for (v) |*w, j|
            w.* = @intToFloat(f32, j) + @intToFloat(f32, i);
        x.* = v;
    }
    const allocator = std.testing.allocator;
    var key = random.PRNGKey(random.Hashes.aes5){.seed = 42};
    var db = try Database(V).init(allocator, key, data[0..]);
    defer db.deinit();
    var query = data[12] + @splat(100, @as(f32, 0.1));
    var enc_query = db.encode(query);
    var all = try db.apply(allocator, enc_query);
    defer allocator.free(all);
    std.debug.print("{any}\n", .{all});
}
