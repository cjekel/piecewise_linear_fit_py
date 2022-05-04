import struct
import ctypes
import numpy as np

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def float2hex(s):
    fp = ctypes.pointer(ctypes.c_float(s))
    cp = ctypes.cast(fp,ctypes.POINTER(ctypes.c_long))
    return hex(cp.contents.value)


def hex_to_float(h):
    i = int(h,16)
    return struct.unpack('<f',struct.pack('<I', i))[0]


def hex2float(h):
    i = int(h,16)
    cp = ctypes.pointer(ctypes.c_int(i))
    fp = ctypes.cast(cp,ctypes.POINTER(ctypes.c_float))
    return fp.contents.value


if __name__ == '__main__':
    # f = [-1.2, 17.5, 2.88, -2.4]
    # h = []
    # for i in f:
    #     print(float_to_hex(i),"   |   ",float2hex(i))
    #     h.append(float_to_hex(i))
    # print(h)
    # for i in h:
    #     print(hex_to_float(i),"   |   ",hex2float(i))
    a = np.linspace(-8, 8, num=1600, endpoint=False)
    file = open('sigmoidin.txt', 'w')
    for index in range(a.shape[0]):
        file.write(str(float_to_hex(a[index]))[2:])
        file.write('\n')
    file.write(str(float_to_hex(7.999755859375))[2:])
    file.write('\n')
    file.close()

# a = '3f800000'
# print(type(a), hex_to_float(a))
