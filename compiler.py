#!/usr/bin/env python3

from dataclasses import dataclass
from functools import singledispatch
from collections.abc import Iterable

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
BLOCK_BEGIN = NilToken('block begin')
BLOCK_END = NilToken('block end')

### Parser combinator

@dataclass
class Stream:
    _stream: str
    i: int = 0
    indent: int = 0
    @property
    def stream(self): return self._stream[self.i:]

    @property
    def row(self): return sum((1 for c in self._stream[:self.i] if c == '\n'))

    @property
    def col(self): return next((j for j in range(self.i) if self._stream[self.i - j] == '\n'), self.i)

    def empty(self): return self.i >= len(self._stream)


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

class FunctionCall(tuple):
    @property
    def name(self): return self[0]
    @property
    def args(self): return self[1:]

class Assign(tuple):
    def __new__(cls, a, b): return super(Assign, cls).__new__(cls, tuple(('=', a, b)))
    @property
    def lhs(self): return self[0]
    @property
    def rhs(self): return self[1]

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

def indentation():
    p = many(char(" "))
    def indentationf(stream):
        c, new_stream = p(stream)
        if type(c) == ParseError:
            return c, stream
        if len(c) == stream.indent:
            return EMPTY, new_stream
        elif len(c) > stream.indent:
            stream.indent = len(c)
            return BLOCK_BEGIN, new_stream
        elif len(c) < stream.indent:
            stream.indent = len(c)
            return BLOCK_END, new_stream
    return indentationf

# we don't care about space, so we discard it
# <space> := (' ' | '\t')*
space = discard(many(char(" \t")))
# <newline> := '\n'
newline = discard(char("\n"))
# <number> := <digit> <digit>*
number = convert(one_or_more(digit), lambda x: int("".join(x)))
# <identifier> := <alpha> (<alphanumeric> | '-' | '_')*
identifier = convert(seq(alpha, many(oneof(alphanumeric, char("-_")))), lambda x: "".join([x[0]] + x[1]))
# <function-call> := <identifier> '(' <expr> (',' <space> <expr>)* ')'
function_call = convert(
        seq(identifier, discard(char("(")), intersperse(lambda x: expr(x), discard(seq(char(","), space))), discard(char(")"))),
        lambda x: ["call"] + x)

# <expr> := <number> | <function-call> | <identifier>
# NOTE potential ambiguity of function-call vs identifier since they both start with an identifier
expr = oneof(number, function_call, identifier) # NOTE: order matters here (can't do identifier, function call)
# <assign-stmt-body> := <identifier> <space> '=' <space> <expr>
assign_stmt_body = convert(seq(identifier, space, discard(char("=")), space, expr), lambda x: ["=", x[0], x[1]])
# <return-stmt-body> := 'return' <space> <expr>
return_stmt_body = seq('return', space, expr)

if_stmt = seq('if', space, expr, discard(char(':')),
        lambda x: block(x),
        'else', discard(char(':')),
        lambda x: block(x))

# <stmt> := <indentation> (<return-stmt-body> | <assign-stmt-body | <if-stmt> | <expr>) <newline>
stmt = convert(seq(discard(indentation()), oneof(return_stmt_body, assign_stmt_body, expr), newline), lambda x: x[0])
# <block> := newline, BLOCK_BEGIN (<stmt>)+ BLOCK_END
block = convert(seq(discard(newline), one_or_more(stmt)), lambda x: x[0])
# <function> := 'def' <space> <identifier> '(' (<identifier> (',' <space> <identifier>)*) ')' ':' <newline> <block>
function = seq('def', space, identifier, char('('), intersperse(identifier, discard(seq(char(','), space))), char(')'), char(':'), block)

### A-normal form

gensym_counter = 0
def gensym():
    global gensym_counter
    gensym_counter += 1
    return f'tmp{gensym_counter}'

def is_trivial(b):
    return type(b) in {str, int, float}


@singledispatch
def normalize_stmt(stmt): # expr case
    norm, hoisted = normalize_expr(stmt)
    return hoisted.add(norm)

@singledispatch
def normalize_expr(node):
    return node, Block()

@normalize_expr.register(FunctionCall)
def _(node: FunctionCall):
    new_args = []
    hoisted = Block()
    for arg in node.args:
        a, h_ = normalize_expr(arg)
        hoisted.add(h_)
        if is_trivial(a):
            new_args.append(a)
        else:
            new_var = gensym()
            hoisted.add(Assign(new_var, a))
            new_args.append(new_var)
    return FunctionCall([node.name] + new_args), hoisted

### Code gen

#call_registers = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
#def emit_function_call(fcall):
#    assert fcall[0] == 'call'
#    _, function_name, args = fcall


if __name__ == "__main__":
    import doctest

    doctest.testmod()
