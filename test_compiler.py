#!/usr/bin/env python3
import unittest
from compiler import *

class TestCompiler(unittest.TestCase):
    def test_stream(self):
        s = Stream('''Some
text
on some lines''')
        self.assertEqual(s.row, 0)
        self.assertEqual(s.col, 0)

        s.i = 6
        self.assertEqual(s.row, 1)
        self.assertEqual(s.col, 2)

    def test_identifier(self):
        self.assertEqual(parse('whee whoo', identifier), 'whee')
        self.assertEqual(parse('wh3e whoo', identifier), 'wh3e')

    def test_parse_number(self):
        self.assertEqual(parse('32 45', number), 32)
        self.assertEqual(parse('007', number), 7)
        self.assertEqual(parse('0x13', number), 0)

    def test_expr(self):
        self.assertEqual(parse('a=b', identifier), 'a')
        self.assertEqual(parse('a=b', expr), 'a')

    def test_stmt(self):
        self.assertEqual(parse('a=b\n', stmt), ['=', 'a', 'b'])
        self.assertEqual(parse('a(b)\n', stmt), ['call', 'a', ['b']])

    def test_function_call(self):
        self.assertEqual(parse('f(1)', function_call), ['call', 'f', [1]])
        self.assertEqual(parse('f(2, x)', function_call), ['call', 'f', [2, 'x']])
        self.assertEqual(parse('fcall(g(x), 2)', function_call), ['call', 'fcall', [['call', 'g', ['x']], 2]])

    def test_assign_stmt(self):
        self.assertEqual(parse('a = b', assign_stmt_body), ['=', 'a', 'b'])
        self.assertEqual(parse('a = f(b, 1)', assign_stmt_body), ['=', 'a', ['call', 'f', ['b', 1]]])

    def test_return_stmt(self):
        self.assertEqual(parse('return 3', return_stmt_body), ['return', 3])
        self.assertEqual(parse('return f(3, 4)', return_stmt_body), ['return', ['call', 'f', [3, 4]]])

    def test_parse_identifier(self):
        with self.assertRaises(Exception) as e:
            parse('32 45', identifier)
        self.assertTrue(str(e.exception).startswith('1:1: expected one of'), f"got: '{e.exception}'")

        self.assertEqual(parse('l33t c0d3r', identifier), 'l33t')
        self.assertEqual(parse('val+34', identifier), 'val')

    def test_if(self):
        self.assertEqual(parse(
'''if true:
    print(2)
else:
    return 1
''', if_stmt), ['if', 'true', [['call', 'print', [2]]], 'else', [['return', 1]]])

    def test_block(self):
        b1 = Block('a', 'b', 'c')
        self.assertEqual(b1, ['a', 'b', 'c'])

        b2 = Block('a', Block('b'), 'c')
        self.assertEqual(b2, ['a', 'b', 'c'])

    def test_normalize(self):
        tree1 = FunctionCall(('a', 'b', 'c'))
        self.assertEqual(normalize(tree1)[0], ('a', 'b', 'c'))

        global gensym_counter
        gensym_counter = 0

        tree2 = FunctionCall(('a', FunctionCall(('b')), 'c'))
        n = normalize(tree2)
        self.assertEqual(n[1].add(n[0]),
                [('=', 'tmp1', ('b',)),
                 ('a', 'tmp1', 'c')])

        gensym_counter = 0
        tree2 = FunctionCall(('a', FunctionCall(('b', FunctionCall(('x',)))), 'c'))
        n = normalize(tree2)
        self.assertEqual(n[1].add(n[0]),
                [('=', 'tmp2', ('x',)),
                 ('=', 'tmp3', ('b', 'tmp2')),
                 ('a', 'tmp3', 'c')])


if __name__ == '__main__':
    unittest.main()
