#!/usr/bin/env python3
import unittest
from compiler import *

class TestCompiler(unittest.TestCase):
    def test_identifier(self):
        self.assertEqual(parse('whee whoo', identifier), 'whee')
        self.assertEqual(parse('wh3e whoo', identifier), 'wh3e')

    def test_parse_number(self):
        self.assertEqual(parse('32 45', number), 32)
        self.assertEqual(parse('007', number), 7)
        self.assertEqual(parse('0x13', number), 0)

    def test_seq(self):
        self.assertEqual(parse('w0t', oneof(seq(alpha, alpha), seq(alpha, digit, alpha))), ['w', '0', 't'])

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
        self.assertTrue(str(e.exception).startswith('1:1: expected one of'))

        self.assertEqual(parse('l33t c0d3r', identifier), 'l33t')
        self.assertEqual(parse('val+34', identifier), 'val')

    def test_if(self):
        self.assertEqual(parse('''if true:
    print(2)
else:
    return 1''', if_stmt), ['if', 'true', [['print', 2]], 'else', [['return', 1]]])

if __name__ == '__main__':
    unittest.main()
