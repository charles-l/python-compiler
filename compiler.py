#!/usr/bin/env python3
# yank the following, and execute :@"
# set makeprg=python3\ ./compiler.py\ &&\ python3\ ./test_compiler.py

from dataclasses import dataclass
from functools import singledispatch
from operator import itemgetter
from typing import Callable, Tuple, List, Union
import itertools

class NilToken(str):
    '''
    NilTokens are used to represent bits of information about the token stream
    to change the structure of the resulting tree. They act like normal strings
    in almost every way. The only two differences are the functionality for repr()
    and their internal reference.

    >>> a = NilToken('a')
    >>> a + 'some str' # no effect on normal strings
    'some str'
    >>> str(a)
    ''
    >>> empty = ''
    >>> # Python does string interning by default
    >>> # Empty and '' reference the same memory location
    >>> empty is ''
    True
    >>> # But since we constructed the NilToken instance, it has a new memory location
    >>> a is ''
    False
    >>> a is a # The only way to get 'is' to evaluate to true is to use the exact same reference
    True
    >>> b = NilToken('b')
    >>> a is b
    False
    >>> a.name # for debug
    '<a>'
    '''
    def __new__(cls, name):
        o = str.__new__(cls, '')
        o.name = '<' + name + '>'
        return o

# EMPTY is not ''
# EMPTY == ''
EMPTY = NilToken('empty')
INDENT = NilToken('indent')
DEDENT = NilToken('dedent')

### Parser combinator

@dataclass
class Stream:
    _stream: str
    i: int = 0
    indent: int = 0
    stream = property(lambda self: self._stream[self.i:])
    row = property(lambda self: sum((1 for c in self._stream[:self.i] if c == '\n')))
    col = property(lambda self: next((j for j in range(self.i) if self._stream[self.i - j] == '\n'), self.i))
    def empty(self): return self.i >= len(self._stream)

ParserT = Callable[[Stream], Tuple[object, Stream]]

@dataclass
class ParseError:
    stream: Stream # TODO: add an index, use a slice instead of updating `stream` so we can peek backwards for error messages
    expected: str  # TODO: remove expected. use stream location instead (?)
    got: str
    many: bool = False

    def error_string(self):
        if self.many:
            self.expected = f"one of: {self.expected}"
        if self.got:
            return f"{self.stream.row + 1}:{self.stream.col + 1}: expected {self.expected} but got {self.got}"
        else:
            return f"{self.stream.row + 1}:{self.stream.col + 1}: expected {self.expected}"

def empty(s: Stream):
    """
    Parser.

    The trival case. Parses an empty string successfully.
    >>> parse('nothing to see here, move along', empty)
    ''
    """
    return EMPTY, s

# TODO: combine this with char()
# TODO: make this part of the stream object
def next_char(s: Stream):
    """
    The lowest level parser.
    Return the next char in the string and the advanced Stream.
    >>> s = Stream('some string')
    >>> c1, s1 = next_char(s)
    >>> c1
    's'
    >>> c2, s2 = next_char(s1)
    >>> c2
    'o'
    >>> s2.row, s2.col
    (0, 2)

    It should also handle newlines correctly:
    >>> s = Stream("some\\nstring")
    >>> for i in range(6):
    ...     c, s = next_char(s)
    >>> c
    's'
    >>> s.row, s.col
    (1, 2)

    Empty streams return a ParseError. This will be useful later.
    >>> s = Stream('')
    >>> e, s = next_char(s)
    >>> e.got
    'EOF'
    """
    if s.empty():
        c = ParseError(s, "<any char>", "EOF")
    else:
        c = s.stream[0]
    if c == "\n":
        return c, Stream(s._stream, s.i + 1)
    else:
        return c, Stream(s._stream, s.i + 1)

def parse(s: str, parser):
    """
    Execute the parser on a specific string
    """
    res, stream = parser(Stream(s))
    if type(res) == ParseError:
        raise Exception(res.error_string())
    return res

def char(expected=None):
    """
    Return a parser that expects a specific character
    >>> parse('a word', char())
    'a'
    >>> parse('a word', char('a'))
    'a'
    >>> parse('the word', char('a'))
    Traceback (most recent call last):
        ...
    Exception: 1:1: expected 'a' but got 't'
    >>> parse('the word', char('at')) # 'a' or 't'
    't'
    """
    if expected == None:  # any character
        return next_char

    def charf(stream):
        c, new_stream = next_char(stream)
        if type(c) == ParseError:
            return ParseError(stream, repr(expected), c.got, many=len(expected) > 1), new_stream
        if c not in expected:
            return ParseError(stream, repr(expected), repr(c), many=len(expected) > 1), new_stream
        else:
            return (c, new_stream)

    return charf


alpha = char("abcdefghijklmnopqrstuvwxyz")
digit = char("1234567890")

def oneof(*ps):
    """
    Combinator.
    Expect one of the parsers to parse (and return the result of the first
    one that does)
    >>> alphanumeric = oneof(alpha, digit) # equivalent to char('abcdefghijklmnopqrstuvwxyz1234567890')
    >>> parse('wheee', alphanumeric)
    'w'
    >>> parse('32', alphanumeric)
    '3'
    >>> parse('-', oneof(char('a'), char('b')))
    Traceback (most recent call last):
        ...
    Exception: 1:1: expected one of: 'a', 'b'
    """
    def oneoff(stream):
        errs = []
        for p in ps:
            v, stream1 = p(stream)
            if type(v) == ParseError:
                errs.append(v)  # collect the errors for later (if all parsers fail)
            else:
                return (v, stream1)
        else:
            return ParseError(stream, ", ".join([x.expected for x in errs]), None, many=True), stream
    return oneoff

alphanumeric = oneof(alpha, digit)

def seq(*ps):
    """
    Combinator.
    Expects all of the parsers to parse in sequence (returning the accumulated
    result of all of them)
    >>> parse('a2b', seq(alpha, digit, alpha))
    ['a', '2', 'b']
    >>> parse('acb', seq(alpha, digit, alpha))
    Traceback (most recent call last):
        ...
    Exception: 1:2: expected one of: '1234567890' but got 'c'
    """
    def seqf(stream):
        a = []
        s = stream
        for p in ps:
            if type(p) == str:
                e = p
                p = expect(identifier, e)
            v, new_stream = p(s)
            if type(v) == ParseError:
                return (v, stream)
            else:
                if v is not EMPTY:
                    a.append(v)
            s = new_stream  # advance stream
        return (a, s)
    return seqf

def convert(p, convert_f):
    """
    Combinator, I guess?? More of a utility function than anything else.
    Collect the results of a combinator or parser and convert it into a different type.
    >>> parse('123', many(digit))
    ['1', '2', '3']
    >>> parse('123', convert(many(digit), lambda x: int(''.join(x))))
    123
    """
    def convertf(stream):
        val, new_stream = p(stream)
        if type(val) == ParseError:
            return val, stream  # pass errors through
        return convert_f(val), new_stream
    return convertf

def many(p):
    """
    Combinator.
    Repeatedly apply `p` and collect the result.
    >>> parse('never gonna give you up, never gonna let you down', many(alpha))
    ['n', 'e', 'v', 'e', 'r']
    >>> parse('13', many(digit))
    ['1', '3']
    """
    def manyf(stream):
        a = []
        while True:
            val, new_stream = p(stream)
            if type(val) == ParseError:  # break on the first error
                return (a, stream)
            else:
                a.append(val)
            stream = new_stream  # advance stream
        return a, stream
    return manyf

def one_or_more(p):
    """
    Combinator.
    Like many, but expects at least one:
    >>> parse('wow', many(alpha)) == parse('wow', one_or_more(alpha))
    True
    >>> parse('w', many(alpha)) == parse('w', one_or_more(alpha))
    True
    >>> parse('', many(alpha))
    []
    >>> parse('', one_or_more(alpha))
    Traceback (most recent call last):
        ...
    Exception: 1:1: expected one of: 'abcdefghijklmnopqrstuvwxyz' but got EOF
    """
    return convert(seq(p, many(p)), lambda x: [x[0]] + x[1])

def one_or_none(p):
    """
    Combinator.

    Try to get a `p` or nothing.
    >>> parse('whoop', one_or_none(alpha))
    'w'
    >>> parse('whoop', one_or_none(digit))
    ''
    """
    return oneof(p, empty)

def discard(p):
    """
    Combinator.

    Throws away a parser result if it succeeded.
    >>> parse('whoop', discard(alpha))
    ''
    >>> space = discard(many(char(' \t')))
    >>> parse('   whee', seq(space, alpha))
    ['w']
    """
    def discardf(stream):
        val, new_stream = p(stream)
        if type(val) == ParseError:
            return val, stream
        else:
            return EMPTY, new_stream
    return discardf

# TODO: remove expectations in other parts of the code
# TODO: write expect_predicate(p, predicate, error_message)
# TODO: write "bind" or something that allows simpler pipelining for errors
def expect(p, expected_value):
    """
    Utility function.
    Expects `p` to parse an exact value. Just passes the parse result through on success.
    >>> if_kwd = expect(convert(many(alpha), lambda x: ''.join(x)), 'if')
    >>> parse('if true', if_kwd)
    'if'
    >>> parse('else:', if_kwd)
    Traceback (most recent call last):
        ...
    Exception: 1:1: expected 'if' but got 'else'
    """
    def matchf(stream):
        val, new_stream = p(stream)
        if type(val) == ParseError:
            return ParseError(stream, repr(expected_value), val.got), stream
        if val != expected_value:
            return ParseError(stream, repr(expected_value), repr(val)), stream
        else:
            return val, new_stream
    return matchf

### Language parser

class Block(list):
    def __init__(self, *nodes):
        l = []
        for n in nodes:
            if type(n) == Block:
                l.extend(n)
            else:
                l.append(n)
        self.extend(l)

    def add(self, b):
        if type(b) == Block:
            self.extend(b)
        else:
            self.append(b)
        return self

# took inspiration from collections.namedtuple:
# https://github.com/python/cpython/blob/58ccd201fa74287ca9293c03136fcf1e19800ef9/Lib/collections/__init__.py#L290
def nodeclass(name, fields, hole_values=[]):
    fields = fields.replace(',', ' ').split() if type(fields) == str else fields
    class_namespace = {}
    for i, f in enumerate(fields):
        if f == '_':
            assert i < len(hole_values), f"default value for {i} is not passed!"
            continue
        elif f.startswith('*'):
            assert i == len(fields) - 1, "splat arg must be last field"
            class_namespace[f.lstrip('*')] = property(lambda self: self[i:], doc=f'alias for elements [{i}:]')
        else:
            class_namespace[f] = property(itemgetter(i), doc=f'alias for element at {i}')

    def __new__(cls, *args):
        for i, f in enumerate(fields):
            if f == '_': args = args[:i] + (hole_values[i],) + args[i:]
        return tuple.__new__(cls, args)

    class_namespace['__new__'] = __new__

    return type(name, (tuple,), class_namespace)

FunctionCall = nodeclass('FunctionCall', 'name *args')
Assign = nodeclass('Assign', '_ lhs rhs', ['='])
If = nodeclass('If', '_ cond then otherwise', ['if'])
Return = nodeclass('Return', '_ value', ['return'])

def intersperse(p, delimp):
    """
    Combinator.
    Expects one or more `p`s to be intersperesed by `delimp`
    >>> parse('a,b,c', intersperse(alpha, char(',')))
    ['a', ',', 'b', ',', 'c']
    >>> parse('a, b,  c', intersperse(alpha, discard(seq(char(','), many(char(' '))))))
    ['a', 'b', 'c']
    """
    return convert(seq(p, many(seq(delimp, p))), lambda x: [x[0]] + sum(x[1], []))

def indentation(expect='same'):
    p = many(char(" "))
    def indentationf(stream):
        c, new_stream = p(stream)
        if type(c) == ParseError:
            return c, stream

        if len(c) == stream.indent:
            actual = 'same'
        elif len(c) > stream.indent:
            actual = 'indent'
        elif len(c) < stream.indent:
            actual = 'dedent'

        if expect == actual:
            new_stream.indent = len(c)
            return EMPTY, new_stream

        return ParseError(stream, f'indentation level: {expect} (from {stream.indent})', f'indentation level: {actual} ({len(c)})'), stream

    return indentationf

# we don't care about space, so we discard it
# <space> := (' ' | '\t')*
space: ParserT = discard(many(char(" \t")))
# <newline> := '\n'
newline: ParserT = discard(char("\n"))
# <number> := <digit> <digit>*
number: ParserT = convert(one_or_more(digit), lambda x: int("".join(x)))
# <identifier> := <alpha> (<alphanumeric> | '-' | '_')*
identifier: ParserT = convert(seq(alpha, many(oneof(alphanumeric, char("-_")))), lambda x: "".join([x[0]] + x[1]))
# <function-call> := <identifier> '(' <expr> (',' <space> <expr>)* ')'
function_call: ParserT = convert(
        seq(identifier, discard(char("(")), intersperse(lambda x: expr(x), discard(seq(char(","), space))), discard(char(")"))),
        lambda x: ["call"] + x)

# <expr> := <number> | <function-call> | <identifier>
# NOTE potential ambiguity of function-call vs identifier since they both start with an identifier
expr: ParserT = oneof(number, function_call, identifier) # NOTE: order matters here (can't do identifier, function call)
# <assign-stmt-body> := <identifier> <space> '=' <space> <expr>
assign_stmt_body: ParserT = convert(seq(identifier, space, discard(char("=")), space, expr), lambda x: ["=", x[0], x[1]])
# <return-stmt-body> := 'return' <space> <expr>
return_stmt_body: ParserT = seq('return', space, expr)

if_stmt: ParserT = seq('if', space, expr, discard(char(':')),
        lambda x: block(x),
        indentation('same'), 'else', discard(char(':')),
        lambda x: block(x))

# <stmt> := (<if-stmt> | <return-stmt-body> | <assign-stmt-body | <expr>) <newline>
stmt: ParserT = convert(seq(oneof(if_stmt, return_stmt_body, assign_stmt_body, expr), newline), lambda x: x[0])
# <block> := <newline> (<indent> <stmt>)+
block: ParserT = convert(seq(newline, one_or_more(convert(seq(indentation('indent'), stmt), lambda x: x[0]))), lambda x: x[0])
#import tmp; block = tmp.trace_function(block)
# <function> := 'def' <space> <identifier> '(' (<identifier> (',' <space> <identifier>)*) ')' ':' <newline> <block>
function: ParserT = seq('def', space, identifier, char('('), intersperse(identifier, discard(seq(char(','), space))), char(')'), char(':'), block)

### A-normal form normalizer

gensym_counter = 0
def gensym(prefix='tmp'):
    global gensym_counter
    gensym_counter += 1
    return f'{prefix}{gensym_counter}'

def is_trivial(b):
    return type(b) in {str, int, float}

@singledispatch
def normalize_stmt(stmt): # expr case
    norm, hoisted = normalize_expr(stmt)
    return hoisted.add(norm)

@normalize_stmt.register(Block)
def _(block):
    norm_block = Block()
    for b in block:
        norm_block.add(normalize_stmt(b))
    return norm_block

@normalize_stmt.register(If) # type: ignore
def _(ifnode):
    norm_cond, hoisted = normalize_expr(ifnode.cond)
    norm_then = normalize_stmt(ifnode.then)
    norm_otherwise = normalize_stmt(ifnode.otherwise)
    norm_if = If(maybe_hoist(norm_cond, hoisted), normalize_stmt(ifnode.then), normalize_stmt(ifnode.otherwise))
    if hoisted:
        return hoisted.add(norm_if)
    else:
        return norm_if

@normalize_stmt.register(Return) # type: ignore
def _(ret):
    n, hoisted = normalize_expr(ret.value)
    if hoisted:
        return hoisted.add(Return(maybe_hoist(n, hoisted)))
    else:
        return n

@normalize_stmt.register(Assign) # type: ignore
def _(assign):
    n, hoisted = normalize_expr(assign.rhs)
    norm_a = Assign(assign.lhs, n)
    if hoisted:
        return hoisted.add(norm_a)
    else:
        return norm_a

@singledispatch
def normalize_expr(node):
    return node, Block()

def maybe_hoist(expr, hoisted):
    if is_trivial(expr):
        return expr
    new_var = gensym()
    hoisted.add(Assign(new_var, expr))
    return new_var

@normalize_expr.register(FunctionCall) # type: ignore
def _(function_call):
    new_args = []
    hoisted = Block()
    for arg in function_call.args:
        a, h_ = normalize_expr(arg)
        hoisted.add(h_)
        new_args.append(maybe_hoist(a, hoisted))
    return FunctionCall(function_call.name, *new_args), hoisted

### Codegen (x86-64)
# TODO: register allocation -- go through the tree and tag bindings with registers or memory locations
import struct

def pack8(imm, signed=False):
    return struct.pack('b' if signed else 'B', imm)

def pack16(imm):
    return struct.pack('H', imm)

def pack32(imm):
    return struct.pack('<L', imm)

def pack64(imm):
    return struct.pack('<Q', imm)

def reg64_p(x): return x in regs64
def reg32_p(x): return x in regs32
def label_p(x): return type(x) == str

def get_reg_p(reg):
    if reg64_p(reg): return reg64_p
    if reg32_p(reg): return reg32_p

def get_reg_t(reg):
    if reg64_p(reg): return regs64
    if reg32_p(reg): return regs32

def general_purpose_reg64(x): return reg64_p(x) and regs64[x] < 8
def mem_p(reg_p):
    assert(reg_p in (reg64_p, reg32_p))
    return lambda x: type(x) in (list, tuple) and reg_p(x[0])
def or_p(a, b): return lambda x: a(x) or b(x)
def imm32_p(x): return type(x) == int
def imm8_p(x): return type(x) == int and 0 <= x <= 255

_modrm_pattern = {'*': 0b00, '*+disp8': 0b01, '*+disp32': 0b10, 'direct': 0b11}
def _pack_modrm(reg_id, rm, mod):
    '''
    Construct a ModR/M byte

    bit pattern:
    xx xxx xxx = mod (2 bits) | reg (3 bits) | rm (3 bits)
    '''
    return struct.pack('B', _modrm_pattern[mod] << 6 | reg_id << 3 | rm)

def modrm(reg1, reg2_or_mem):
    '''
    Build a ModR/M byte sequence (used to encode register/memory arguments efficiently for a x86)

    Example 1.

            mov modrm('eax', 'ecx')                  # ecx = eax

        To emit an instruction to move the contents of eax into ecx, we want to encode the ModR/M byte
        using "direct" addressing (i.e. no memory offsets -- just copy one register directly into the other).
        We would encode this into a two byte sequence:

            [byte 1  (mov opcode)] '1011001'
            [byte 2 (ModR/M byte)] '11' (direct mod), '000' (eax), '001' (ecx)

        `[byte 2]` is the "argument" to the mov opcode. `mod` is kind of like a flag that changes the way `mov`
        can work. When set to '11' it uses direct mode.

    Example 2.

            mov modrm('eax', ['ecx'])                # ecx = *eax

        In this situation, we actually want to treat the value in eax as a pointer. Instead of eax's its value
        to ecx, we want to dereference it, and copy the value at the memory location to ecx. This is called
        'indirect' addressing, since we're doing pointer dereference (indirection) to access the value.

            [byte 1  (mov opcode)] '1011001'
            [byte 2 (ModR/M byte)] '00' (indirect mod), '000' (eax), '001' (ecx)

        The ModR/M byte is only 1 bit different than the previous example, but performs a very different
        function.

    Example 3.
        If we want to emit an instruction sequence to perform the following:

            mov modrm('eax', ['edx', 4]) edx = *(eax + 4)

        Which may be more recognizable in the following form:

            edx = eax[4]

        We want to take the memory address in eax, add 4 to it, and get the value there. We could emit a
        sequence of instructions to essentially perform:

            ecx = eax
            ecx += 4
            edx = *ecx

        However, dereferencing a register plus a known offset is such a common operation that having to emit
        3 instructions every time this came up would lead to inefficiency and code bloat. That's why the ModR/M
        byte has two more mod settings. When it's set to '01', it'll perform a 8-bit indirect dereference,
        and when set to '10' it'll perform a 32-bit indirect dereference. The only reason there are two

        In this case we want to emit the ModR/M byte, followed by the sequence of bytes encoding the offset.
        The function will emit:

            [byte 1  (mov opcode)] '1011001'
            [byte 2 (ModR/M byte)] '01' (indirect + disp8 mod), '000' (eax), '010' (edx)
            [byte 3       (disp8)] '0000100' (offset)


    For info: https://wiki.osdev.org/X86-64_Instruction_Encoding#ModR.2FM_and_SIB_bytes
    '''
    reg_p = get_reg_p(reg1)
    regs = get_reg_t(reg1)
    if mem_p(reg_p)(reg2_or_mem):
        reg2 = reg2_or_mem[0]
        offset = 0 if len(reg2_or_mem) == 1 else reg2_or_mem[1]
        if not offset: # *reg2
            return _pack_modrm(regs[reg1], regs[reg2], 'indirect')
        elif imm8_p(offset): # *(reg2 + offset) when offset is small enough to be encoded in 1 byte
            return _pack_modrm(regs[reg1], regs[reg2], '*+disp8') + struct.pack('B', offset)
        elif imm32_p(offset): # *(reg2 + offset) when offset must be encoded in 4 bytes
            return _pack_modrm(regs[reg1], regs[reg2], '*+disp32') + struct.pack('<L', offset)
        assert False, f'Unknown indirect addressing mode {reg2_or_mem}'
    else:
        return _pack_modrm(regs[reg1], regs[reg2_or_mem], 'direct')

import io
codegen_buf = io.BytesIO()

## Assembler
'''
Note [x86-64 instruction encoding]:
===================================

x86-64 opcodes are encoded according to the following pattern:
    <opcode> := <legacy_prefix (1-4 bytes)>?
                <opcode_with_prefix (1-4 bytes)>
                <ModR/M (1 byte)>?
                <SIB (1 byte)>?
                <displacement (1, 2, 4, or 8 bytes)>?
                <immediate (1, 2, 4, or 8 bytes)>?

The legacy prefix or opcode prefix just change the behavior of the opcode (they modify which
addressing mode is used (e.g. 32 or 64 bit), allow use of extended registers or just make more
opcodes available to use)

If an code utilizes an immediate value, it will require the little endian encoded value to be included.

ModR/M bytes encode registers/memory offsets for instructions. If a memory offset is used
in the ModR/M byte, then displacement bytes will be required. See modrm() for more info.

TODO: explain SIB bytes

See https://wiki.osdev.org/X86-64_Instruction_Encoding for more information.
'''

# Registers are encoded by index, see Note [x86-64 instruction encoding] for more info
regs64 = {r: i for i, r in enumerate('rax rcx rdx rbx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15'.split())}
regs32 = {r: i for i, r in enumerate('eax ecx edx ebx esp ebp esi edi r8d r9d r10d r11d r12d r13d r14d r15d'.split())}

def compute_offset(label):
    def f(loc):
        if label not in labels:
            raise Exception('undefined label', label)
        return labels[label] - loc - 2
    return f

def pass2(instructions: List[Union[bytes, tuple]]) -> bytes:
    '''
    The second pass of the assembler that fills in jump locations and combines the final
    instruction bytestream
    '''
    r = b''
    for i in instructions:
        if isinstance(i, tuple):
            bytelen, f = i
            b = f(len(r))
            assert(len(b) == bytelen) # sanity check
            r += b
        elif isinstance(i, bytes):
            r += i
        else:
            assert False, f'{type(i)} not handled'
    return r

def write_elf(text_section: bytes) -> bytes:
    '''
    for more info see:
    * https://www.muppetlabs.com/~breadbox/software/tiny/teensy.html
    * https://cirosantilli.com/elf-hello-world
    * `man elf`
    '''

    entry_vaddr = 0x401000
    #                          magic        class  data           version  abi version  padding
    #                                       elf64  little endian  1
    ident = struct.pack('16b', *b'\x7fELF', 2,     1,             1,       0,           *([0] * 8))
    ehsize = 64
    phentsize = 56
    fsize = ehsize + phentsize + len(text_section)
    #             ident   type       machine     version    entry
    #                     exec       x86_64      1
    elf_header = [ident, pack16(2), pack16(62), pack32(1), pack64(entry_vaddr + ehsize + phentsize)]
    #              phoff                          shoff
    elf_header += [pack64(ehsize), pack64(0)]
    #              flags       ehsize                 phentsize
    #              none        size of this header
    elf_header += [pack32(0),  pack16(ehsize),        pack16(phentsize)]
    #              phnum      shentsize  shnum      shstrndx
    elf_header += [pack16(1), pack16(0), pack16(0), pack16(0)]

    #          type         flags        offset
    #          PT_LOAD      X | R
    pheader = [pack32(1),  pack32(1|4), pack64(0)]
    #           vaddr                paddr                filesize
    pheader += [pack64(entry_vaddr), pack64(entry_vaddr), pack64(fsize)]
    #           textsize                   align
    pheader += [pack64(fsize), pack64(0x1000)]

    return b''.join(elf_header + pheader) + text_section

# a full opcode list can be found here: http://ref.x86asm.net/coder64.html
ops = [
        ## CONTROL FLOW
        (('ret',), lambda _: b'\xc3'),

        # FIXME: calculate distance and emit correct opcode based on whether it's a short or long jump
        (('j', label_p), lambda _, l: (2, lambda x: b'\xeb' + pack8(compute_offset(l)(x), signed=True))),
        (('je', label_p), lambda _, l: (2, lambda x: b'\x74' + pack8(compute_offset(l)(x), signed=True))),
        (('jne', label_p), lambda _, l: (2, lambda x: b'\x75' + pack8(compute_offset(l)(x), signed=True))),

        ## COMPARISONS
        (('cmp', reg32_p, or_p(reg32_p, mem_p(reg32_p))), lambda _, r1, x: b'\x39' + modrm(r1, x)),
        (('cmp', reg64_p, or_p(reg64_p, mem_p(reg64_p))), lambda _, r1, x: b'\x48\x39' + modrm(r1, x)),
        (('cmp', 'eax', imm32_p), lambda _1, _2, x: b'\x3d' + pack32(x)),
        (('cmp', 'rax', imm32_p), lambda _1, _2, x: b'\x48\x3d' + pack32(x)),

        ## MOVS
        # reg -> reg moves get encoded with 0x89 because this is what NASM does. NASM gets used for testing
        # so I did it to be consistent, but someone did a bit more of an analysis here:
        # http://0x5a4d.blogspot.com/2009/12/on-moving-register.html

        ((or_p(reg32_p, mem_p(reg32_p)), '<-', reg32_p), lambda x, _, r1: b'\x89' + modrm(r1, x)),
        # 0x67 prefix for 32-bit address override (see
        ((reg32_p, '<-', mem_p(reg32_p)), lambda r1, _, x: b'\x67\x8b' + modrm(r1, x)),
        # 64 bit movs have a 0x48 prefix to specify 64-bit registers
        ((or_p(reg64_p, mem_p(reg64_p)), '<-', reg64_p), lambda x, _, r1: b'\x48\x89' + modrm(r1, x)),
        ((reg64_p, '<-', mem_p(reg64_p)), lambda r1, _, x: b'\x48\x8b' + modrm(r1, x)),

        ((general_purpose_reg64, '<-', imm32_p), lambda r, _, i: pack8(ord(b'\xb8') + regs64[r]) + pack32(i)),

        ## ARITHMETIC
        (('add', reg32_p, reg32_p), lambda _, r1, r2: b'\x01' + modrm(r2, r1)),

        (('syscall',), lambda _: b'\x0f\x05'),
        (('int', imm32_p), lambda _, x: b'\xcd' + pack8(x))
]

def emit(*args):
    '''
    >>> emit('rax', '<-', 'rcx') == b'\\x48\\x89\\xc8' # binary for mov rcx, rax
    True
    >>> emit('rax <- rcx') == b'\\x48\\x89\\xc8' # cutesy syntax
    True
    '''
    def maybe_int(x):
        try:
            return int(x)
        except:
            return x
    args = list(itertools.chain(*map(lambda x: map(maybe_int, str.split(x)) if type(x) == str else [x], args))) # allow cutsey syntax
    for op, encoder_f in ops:
        if len(op) != len(args):
            continue

        for o, v in zip(op, args):
            if (type(o) == str or type(o) == int) and o != v:
                break
            if callable(o) and not o(v):
                break
        else:
            return encoder_f(*args)
    assert False, f"unknown opcode for {args}"

@singledispatch
def code_gen(node, ctx, g):
    assert False, f'unhandled: {node} type {type(node)}'

@code_gen.register(int) # type: ignore
def _(i: int, ctx, g): # type: ignore
    g.append(emit('rax', '<-', i))

def asmlen(xs: List[Union[bytes, tuple]]) -> int:
    return sum(x[0] if isinstance(x, tuple) else len(x) for x in xs)

labels = {}
def emit_label(name, g):
    labels[name] = asmlen(g)

@code_gen.register(If) # type: ignore
def _(ifstmt, ctx, g):
    false_label = gensym('false')
    end_label = gensym('end')
    code_gen(ifstmt.cond, ctx, g)
    g.append(emit('cmp rax', 0))
    g.append(emit('je', false_label))

    # true branch:
    code_gen(ifstmt.then, ctx, g)
    g.append(emit('j', end_label))

    # false branch:
    emit_label(false_label, g)
    code_gen(ifstmt.otherwise, ctx, g)

    emit_label(end_label, g)

@code_gen.register(Block) # type: ignore
def _(block: Block, ctx, g): # type: ignore
    for stmt in block: code_gen(stmt, ctx, g)

@code_gen.register(Assign) # type: ignore
def _(assign: Assign, ctx, g): # type: ignore
    code_gen(assign.rhs, ctx, g)
    g.append(emit(assign.lhs, '<-', 'rax'))

'''
@code_gen.register(FunctionCall)
def _(f: FunctionCall, ctx, g):
    # TODO: remove this limit and use stack
    assert len(f.args) < len(g.call_registers), 'too many arguments'

    # spill used registers
    for r in g.call_registers[:len(f.args)]: g.push(r)

    for r, a in zip(g.call_registers, f.args):
        g.mov(a, r)

    g.call(f.name)

    # restore spilled registers
    for r in g.call_registers[:len(f.args)][::-1]: g.pop(r)
'''

if __name__ == "__main__":
    import doctest

    doctest.testmod()
