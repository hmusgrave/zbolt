const std = @import("std");
const Allocator = std.mem.Allocator;
const aes = @import("aes.zig");

pub const Hashes = struct {
    aes5: fn (u128) u128 = aes.aes5,
    aes10: fn (u128) u128 = aes.aes10,
}{};

// TODO: Optimize a little
pub fn PRNGKey(comptime mix: fn (u128) u128) type {
    return packed struct {
        seed: u64 align(16),
        gamma: u64 = 0x9e3779b97f4a7c15, // closest odd integer to 1<<64 / phi

        pub fn split(self: *@This(), allocator: Allocator, n: usize) ![]@This() {
            var rtn = try allocator.alloc(@This(), n);
            for (rtn) |*x, i| {
                const new_seed: u128 = self.seed +% self.gamma *% i;
                const new_val = mix((new_seed << 64) + self.gamma);
                x.* = @bitCast(@This(), mix(new_val));
            }
            return rtn;
        }

        fn random(self: *@This(), allocator: Allocator, n: usize) ![]u128 {
            var rtn = try self.split(allocator, n);
            return @ptrCast([*]u128, rtn.ptr)[0..n];
        }

        pub fn uniform(self: *@This(), allocator: Allocator, n: usize) ![]f64 {
            var entropy = try self.random(allocator, (n >> 1) + 1);
            var rtn = @ptrCast([*]f64, entropy.ptr)[0..n];
            var tmp = @ptrCast([*]u64, entropy.ptr)[0..n];
            for (rtn) |*x, i|
                x.* = @intToFloat(f64, tmp[i]) / @intToFloat(f64, std.math.maxInt(u64));
            return rtn;
        }
    };
}

test "Deterministic seed and hash yield deterministic PRNG" {
    const allocator = std.testing.allocator;
    var p = PRNGKey(Hashes.aes5){ .seed = 42 };
    var keys = try p.split(allocator, 3);
    defer allocator.free(keys);
    const a = try keys[0].uniform(allocator, 2);
    defer allocator.free(a);
    const b = try keys[1].uniform(allocator, 3);
    defer allocator.free(b);
    const c = try keys[2].uniform(allocator, 4);
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
