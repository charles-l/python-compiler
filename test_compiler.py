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

    def test_nodeclass(self):
        A = nodeclass('A', 'a,b,c')
        a = A(1,2,3)
        self.assertEqual(a.a, 1)
        self.assertEqual(a.b, 2)
        self.assertEqual(a.c, 3)
        self.assertEqual(a[0], 1)
        self.assertEqual(a[1], 2)
        self.assertEqual(a[2], 3)

        B = nodeclass('B', 'a _ c', [None, 5])
        b = B(1, 3)
        self.assertEqual(b.a, 1)
        self.assertEqual(b.c, 3)
        self.assertEqual(b[0], 1)
        self.assertEqual(b[1], 5)
        self.assertEqual(b[2], 3)

        C = nodeclass('C', 'a b c *z')
        c = C(1,2,3)
        self.assertEqual(c.a, 1)
        self.assertEqual(c.b, 2)
        self.assertEqual(c.c, 3)
        self.assertEqual(c.z, ())
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 2)
        self.assertEqual(c[2], 3)

        D = nodeclass('D', 'a _ c *z', [None, 44])
        d = D(1, 3, 4, 5, 6)
        self.assertEqual(d.a, 1)
        self.assertEqual(d.c, 3)
        self.assertEqual(d.z, (4,5,6))
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 44)
        self.assertEqual(d[2], 3)
        self.assertEqual(d[3], 4)
        self.assertEqual(d[4], 5)
        self.assertEqual(d[5], 6)

    def test_normalize(self):
        tree1 = FunctionCall('a', 'b', 'c')
        self.assertEqual(normalize_expr(tree1)[0], ('a', 'b', 'c'))

        tree2 = FunctionCall('a', FunctionCall('b'), 'c')
        self.assertEqual(normalize_stmt(tree2),
                [('=', 'tmp1', ('b',)),
                 ('a', 'tmp1', 'c')])

        tree3 = FunctionCall('a', FunctionCall('b', FunctionCall('x')), 'c')
        self.assertEqual(normalize_stmt(tree3),
                [('=', 'tmp2', ('x',)),
                 ('=', 'tmp3', ('b', 'tmp2')),
                 ('a', 'tmp3', 'c')])

        tree4 = If(FunctionCall('b', FunctionCall('x')),
                   'c',
                   FunctionCall('d'))
        self.assertEqual(normalize_stmt(tree4),
                [('=', 'tmp4', ('x',)),
                 ('=', 'tmp5', ('b', 'tmp4')),
                 ('if', 'tmp5', ['c'], [('d',)])])

        tree5 = Return(FunctionCall('a', FunctionCall('b')))
        self.assertEqual(normalize_stmt(tree5),
                [('=', 'tmp6', ('b',)),
                 ('=', 'tmp7', ('a', 'tmp6')),
                 ('return', 'tmp7')])

        tree6 = Assign('x', FunctionCall('a', FunctionCall('b')))
        self.assertEqual(normalize_stmt(tree6),
                [('=', 'tmp8', ('b',)),
                 ('=', 'x', ('a', 'tmp8'))])


if __name__ == '__main__':
    unittest.main()
