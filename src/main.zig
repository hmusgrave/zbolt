const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");

const ENDIAN = builtin.target.cpu.arch.endian();

// intel ASM docs: (opt dest), arg1, ..., argn
// zig ASM source: argn, ..., arg1, (opt dest)

inline fn xmm0() @Vector(4, u32) {
    return asm volatile ("": [ret] "={xmm0}" (-> @Vector(4, u32)) :: "xmm0");
}

inline fn xmm1() @Vector(4, u32) {
    return asm volatile ("": [ret] "={xmm1}" (-> @Vector(4, u32)) :: "xmm1");
}

fn debug() void {
    const x = xmm0();
    const y = xmm1();
    std.debug.print("xmm0 {}\n", .{x});
    std.debug.print("xmm1 {}\n", .{y});
}

inline fn aeskeygenassist(temp: @Vector(4, u32), comptime round: u8)
@Vector(4, u32)
{
    return asm volatile ("aeskeygenassist %[round], %[temp], %[ret]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [temp] "x" (temp),
          [round] "i" (round),
    );
}

// TODO: generate this at comptime and write tests
const RCON = [_]u8 {
    0x8d,
    0x01,
    0x02,
    0x04,
    0x08,
    0x10,
    0x20,
    0x40,
    0x80,
    0x1b,
    0x36,
    0x6c,
    0xd8,
    0xab,
    0x4d,
    0x9a,
};

fn keyexpansion(key: []align(4) u8, w: []u32, comptime nw: usize, comptime nk: usize) void {
    var temp: u32 = 0;
    comptime var i: usize = 0;
    inline while (i < nk) : (i += 1) {
        const c = i << 2;
        const data = key[c .. c+4];
        w[i] = @ptrCast(*u32, data).*;
    }
    i = nk;

    inline while (i < nw) : (i += 1) {
        var assist = aeskeygenassist(@Vector(4, u32) {temp, 0, temp, 0},
        RCON[@divFloor(i, nk)]);
        if (i % nk == 0) {
            temp = assist[0];
        } else if (nk > 6 and i % nk == 4) {
            temp = assist[1];
        }
        w[i] = w[i-nk] ^ temp;
    }
}

inline fn aesimc(w: @Vector(4, u32)) @Vector(4, u32) {
    return asm volatile ("aesimc %[w], %[ret]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [w] "x" (w),
    );
}

fn dwexpansion(w: []align(16) u32, dw: []align(16) u32) void {
    for (w) |x,i|
        dw[i] = x;
    var i: usize = 0;
    while (i < w.len) : (i += 4) {
        var wvecptr = @ptrCast(*align(16) [4]u32, dw[i..i+4].ptr);
        @ptrCast(*@Vector(4, u32), dw[i..i+4].ptr).* = aesimc(wvecptr.*);
    }
}

inline fn aesenc(data_state: @Vector(4, u32), round_key: @Vector(4, u32))
@Vector(4, u32)
{
    return asm volatile ("aesenc %[round_key], %[data_state]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [data_state] "{xmm0}" (data_state),
          [round_key] "x" (round_key),
    );
}

inline fn aesenclast(data_state: @Vector(4, u32), round_key: @Vector(4, u32))
@Vector(4, u32)
{
    return asm volatile ("aesenclast %[round_key], %[data_state]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [data_state] "{xmm0}" (data_state),
          [round_key] "x" (round_key),
    );
}

inline fn aesdec(data_state: @Vector(4, u32), round_key: @Vector(4, u32))
@Vector(4, u32)
{
    return asm volatile ("aesdec %[round_key], %[data_state]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [data_state] "{xmm0}" (data_state),
          [round_key] "x" (round_key),
    );
}

inline fn aesdeclast(data_state: @Vector(4, u32), round_key: @Vector(4, u32))
@Vector(4, u32)
{
    return asm volatile ("aesdeclast %[round_key], %[data_state]"
        : [ret] "={xmm0}" (-> @Vector(4, u32))
        : [data_state] "{xmm0}" (data_state),
          [round_key] "x" (round_key),
    );
}

test "basic add functionality" {
    std.debug.print("\n", .{});
    var data = @Vector(4, u32){1, 3, 1, 10};
    var key align(4) = [_] u8 {0} ** (4*1);
    var w: [4*(1+1)] u32 align(16) = undefined;
    var dw: [4*(1+1)] u32 align(16) = undefined;
    keyexpansion(key[0..], w[0..], 8, 1);
    dwexpansion(w[0..], dw[0..]);

    data ^= @bitCast(@Vector(4, u32), w[0..4].*);

    var r1 = aesenc(data, w[4..].*);
    var r2 = aesenclast(r1, w[4..].*);

    r2 ^= @bitCast(@Vector(4, u32), dw[4..].*);

    var d2 = aesdec(r2, dw[0..4].*);
    var d1 = aesdeclast(d2, dw[0..4].*);
    std.debug.print("{any}\n", .{d1});
}
