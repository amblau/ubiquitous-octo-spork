#!/usr/bin/env python3
"""Generates a Hello World ELF64 executable from raw machine code bytes."""

import struct

BASE = 0x400000
ELF_HDR_SIZE = 64
PHDR_SIZE = 56
CODE_OFFSET = ELF_HDR_SIZE + PHDR_SIZE  # 120

# Machine code instructions
code = bytes([
    0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  # mov rax, 1       (sys_write)
    0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,  # mov rdi, 1       (stdout)
    0x48, 0xc7, 0xc6, 0xa2, 0x00, 0x40, 0x00,  # mov rsi, 0x4000a2 (msg addr)
    0x48, 0xc7, 0xc2, 0x0e, 0x00, 0x00, 0x00,  # mov rdx, 14      (msg length)
    0x0f, 0x05,                                  # syscall
    0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00,  # mov rax, 60      (sys_exit)
    0x48, 0x31, 0xff,                            # xor rdi, rdi     (exit code 0)
    0x0f, 0x05,                                  # syscall
])

msg = b"Hello, World!\n"

TOTAL_SIZE = CODE_OFFSET + len(code) + len(msg)  # 176 bytes

# ELF64 header
elf_header = struct.pack(
    "<4sBBBBB7sHHIQQQIHHHHHH",
    b"\x7fELF",          # magic
    2,                    # 64-bit
    1,                    # little-endian
    1,                    # ELF version
    0,                    # OS/ABI
    0,                    # ABI version
    b"\x00" * 7,          # padding
    2,                    # executable
    0x3e,                 # x86-64
    1,                    # ELF version
    BASE + CODE_OFFSET,   # entry point
    ELF_HDR_SIZE,         # program header offset
    0,                    # section header offset
    0,                    # flags
    ELF_HDR_SIZE,         # ELF header size
    PHDR_SIZE,            # program header entry size
    1,                    # program header count
    0,                    # section header entry size
    0,                    # section header count
    0,                    # section name string table index
)

# Program header (PT_LOAD)
program_header = struct.pack(
    "<IIQQQQQQ",
    1,                    # PT_LOAD
    5,                    # PF_R | PF_X
    0,                    # offset
    BASE,                 # virtual address
    BASE,                 # physical address
    TOTAL_SIZE,           # file size
    TOTAL_SIZE,           # memory size
    0x200000,             # alignment
)

with open("hello", "wb") as f:
    f.write(elf_header + program_header + code + msg)

import os
os.chmod("hello", 0o755)

print(f"Built 'hello' ({TOTAL_SIZE} bytes)")
