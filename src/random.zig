const std = @import("std");
const Allocator = std.mem.Allocator;
const aes = @import("aes.zig");

pub const Hashes = struct {
    aes5: fn (u128) u128 = aes.aes5,
    aes10: fn (u128) u128 = aes.aes10,
}{};

// TODO: Optimize a little, add numeric stability
pub fn PRNGKey(comptime mix: fn (u128) u128) type {
    return packed struct {
        seed: u64 align(16),
        gamma: u64 = 0x9e3779b97f4a7c15, // closest odd integer to 1<<64 / phi

        fn split_fill(self: *@This(), buf: []@This()) void {
            for (buf) |*x, i| {
                const new_seed: u128 = self.seed +% self.gamma *% i;
                const new_val = mix((new_seed << 64) + self.gamma);
                x.* = @bitCast(@This(), mix(new_val));
            }
        }

        pub fn split(self: *@This(), comptime n: usize) [n]@This() {
            var rtn: [n]@This() = undefined;
            self.split_fill(rtn[0..]);
            return rtn;
        }

        pub fn split_alloc(self: *@This(), allocator: Allocator, n: usize) ![]@This() {
            var rtn = try allocator.alloc(@This(), n);
            self.split_fill(rtn);
            return rtn;
        }

        fn random(self: *@This(), comptime n: usize) [n]u128 {
            defer self.seed +%= n *% self.gamma;
            return @bitCast([n]u128, self.split(n));
        }

        fn random_alloc(self: *@This(), allocator: Allocator, n: usize) ![]u128 {
            defer self.seed +%= n *% self.gamma;
            var rtn = try self.split_alloc(allocator, n);
            return @ptrCast([*]u128, rtn.ptr)[0..n];
        }

        pub fn randint(self: *@This(), comptime T: type, m: T, M: T, comptime n: usize) [n]T {
            var entropy = self.random(n);
            var rtn: [n]T = undefined;
            const diff = M - m + 1;
            for (rtn) |*x, i|
                x.* = @intCast(T, m + (entropy[i] % diff));
            return rtn;
        }

        pub fn randint_alloc(self: *@This(), allocator: Allocator, comptime T: type, m: T, M: T, n: usize) ![]T {
            var entropy = try self.random_alloc(allocator, n);
            defer allocator.free(entropy);
            var rtn = try allocator.alloc(T, n);
            const diff = M - m + 1; // TODO: Overflow
            for (rtn) |*x, i|
                x.* = @intCast(T, m + (entropy[i] % diff));
            return rtn;
        }

        pub fn uniform(self: *@This(), comptime n: usize) [n]f64 {
            var entropy = self.random((n >> 1) + 1);
            var rtn: [n]f64 = undefined;
            for (rtn) |*x, i| {
                var r = entropy[i >> 1];
                r = if (i & 1 == 0) r & std.math.maxInt(u64) else r >> 64;
                x.* = @intToFloat(f64, r) / @intToFloat(f64, std.math.maxInt(u64));
            }
            return rtn;
        }

        pub fn uniform_alloc(self: *@This(), allocator: Allocator, n: usize) ![]f64 {
            var entropy = try self.random_alloc(allocator, (n >> 1) + 1);
            var rtn = @ptrCast([*]f64, entropy.ptr)[0..n];
            var tmp = @ptrCast([*]u64, entropy.ptr)[0..n];
            for (rtn) |*x, i|
                x.* = @intToFloat(f64, tmp[i]) / @intToFloat(f64, std.math.maxInt(u64));
            return rtn;
        }

        // TODO: reduce to loglinear time
        // TODO: document behavior for zero/neg-weighted data
        pub fn weighted_choice(self: *@This(), comptime T: type, w: []T, comptime n: usize) [n]usize {
            var total: T = 0;
            for (w) |x|
                total += x;
            var entropy = self.uniform(n);
            var rtn: [n]usize = undefined;
            for (entropy) |r, i| {
                var rt: T = w[0];
                var count: usize = 1;
                while (rt < r * total and count < w.len) {
                    rt += w[count];
                    count += 1;
                }
                rtn[i] = count - 1;
            }
            return rtn;
        }
    };
}

test "Weights" {
    var p = PRNGKey(Hashes.aes5){ .seed = 42 };
    var weights = [_]f64{ 1, 5, 25 };
    var random = p.weighted_choice(f64, weights[0..], 100);
    var expected = [_]usize{ 2, 0, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1 };
    try std.testing.expectEqual(random.len, expected.len);
    for (random) |r, i|
        try std.testing.expectEqual(r, expected[i]);
}

test "Deterministic seed and hash yield deterministic PRNG" {
    const allocator = std.testing.allocator;
    var p = PRNGKey(Hashes.aes5){ .seed = 42 };
    var keys = try p.split_alloc(allocator, 3);
    defer allocator.free(keys);
    const a = try keys[0].uniform_alloc(allocator, 2);
    defer allocator.free(a);
    const b = try keys[1].uniform_alloc(allocator, 3);
    defer allocator.free(b);
    const c = try keys[2].uniform_alloc(allocator, 4);
    defer allocator.free(c);
    var target_a = [_]u64{ 4607017741695338843, 4605027437932141949 };
    var target_b = [_]u64{ 4599366927419331910, 4600710989272059620, 4606555051012538015 };
    var target_c = [_]u64{ 4605711111345881508, 4600826801309249528, 4591427475319480601, 4602158710741202426 };
    for (target_a) |x, i| {
        try std.testing.expectEqual(@bitCast(f64, x), a[i]);
        try std.testing.expectEqual(@bitCast(u64, a[i]), x);
    }
    for (target_b) |x, i| {
        try std.testing.expectEqual(@bitCast(f64, x), b[i]);
        try std.testing.expectEqual(@bitCast(u64, b[i]), x);
    }
    for (target_c) |x, i| {
        try std.testing.expectEqual(@bitCast(f64, x), c[i]);
        try std.testing.expectEqual(@bitCast(u64, c[i]), x);
    }
}
