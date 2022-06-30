const V4 = @Vector(4, u32);

const RCON = [_]u8{
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

pub fn expand_key(key: []align(4) u8, comptime nr: usize) [nr + 1]V4 {
    var w: [4 * (nr + 1)]u32 align(16) = undefined;
    var temp: u32 = 0;
    comptime var i: usize = 0;
    inline while (i < nr) : (i += 1) {
        const c = i << 2;
        const data = key[c .. c + 4];
        w[i] = @ptrCast(*u32, data).*;
    }
    i = nr;
    inline while (i < 4 * (nr + 1)) : (i += 1) {
        const assist = asm ("aeskeygenassist %[round], %[temp], %[ret]"
            : [ret] "=x" (-> V4),
            : [temp] "x" (V4{ temp, 0, temp, 0 }),
              [round] "i" (RCON[i / nr]),
        );
        if (i % nr == 0) {
            temp = assist[0];
        } else if (nr > 6 and i % nr == 4) {
            temp = assist[1];
        }
        w[i] = w[i - nr] ^ temp;
    }
    var rtn: [nr + 1]V4 = undefined;
    for (rtn) |*x, k| {
        const c = k << 2;
        x.* = @ptrCast(*V4, w[c .. c + 4].ptr).*;
    }
    return rtn;
}

pub fn aesenc_5(data_state: V4, w: [5]V4) V4 {
    return asm (
        \\vaesenc     %[w1],  %[data], %[ret]
        \\vaesenc     %[w2],  %[ret],  %[ret]
        \\vaesenc     %[w3], %[ret],  %[ret]
        \\vaesenc     %[w4], %[ret],  %[ret]
        \\vaesenclast %[w4], %[ret],  %[ret]
        : [ret] "=&x" (-> V4),
        : [data] "x" (data_state ^ w[0]),
          [w1] "x" (w[1]),
          [w2] "x" (w[2]),
          [w3] "x" (w[3]),
          [w4] "x" (w[4]),
    );
}

pub fn aesenc_10(data_state: V4, w: [10]V4) V4 {
    return asm (
        \\vaesenc     %[w1],  %[data], %[ret]
        \\vaesenc     %[w2],  %[ret],  %[ret]
        \\vaesenc     %[w3], %[ret],  %[ret]
        \\vaesenc     %[w4], %[ret],  %[ret]
        \\vaesenc     %[w5], %[ret],  %[ret]
        \\vaesenc     %[w6], %[ret],  %[ret]
        \\vaesenc     %[w7], %[ret],  %[ret]
        \\vaesenc     %[w8], %[ret],  %[ret]
        \\vaesenc     %[w9], %[ret],  %[ret]
        \\vaesenclast %[w9], %[ret],  %[ret]
        : [ret] "=&x" (-> V4),
        : [data] "x" (data_state ^ w[0]),
          [w1] "x" (w[1]),
          [w2] "x" (w[2]),
          [w3] "x" (w[3]),
          [w4] "x" (w[4]),
          [w5] "x" (w[5]),
          [w6] "x" (w[6]),
          [w7] "x" (w[7]),
          [w8] "x" (w[8]),
          [w9] "x" (w[9]),
    );
}
