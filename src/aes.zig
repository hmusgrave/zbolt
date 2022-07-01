const std = @import("std");
const _aes = @import("_aes.zig");

const V4 = @Vector(4, u32);

var K5 align(4) = [_]u8{0} ** (4 * 4);
var W5: [5]V4 = undefined;

var K10 align(4) = [_]u8{0} ** (4 * 10);
var W10: [10]V4 = undefined;

// TODO: Why can't we create these at comptime?
var _init: bool = false;
pub fn init() void {
    if (!_init) {
        W5 = _aes.expand_key(K5[0..], 4);
        W10 = _aes.expand_key(K10[0..], 9);
        _init = true;
    }
}

pub fn aes5(data: u128) u128 {
    init();
    return @bitCast(V4, _aes.aesenc_5(@bitCast(V4, data), W5));
}

pub fn aes10(data: u128) u128 {
    init();
    return @bitCast(V4, _aes.aesenc_10(@bitCast(V4, data), W10));
}

test "Sample data with trivial key encrypts to known correct values" {
    var data: u128 = 0xFFFF0000FFFF0000FFFF0000FFFF4321;
    try std.testing.expectEqual(aes5(data), 43169916207196930977854297509856771626);
    try std.testing.expectEqual(
        aes10(data),
        19479618102411868804618710722054463414,
    );
}
