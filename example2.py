import LIBLR

grammar = r'''
E: E '+' T | E '-' T | T;
T: T '*' F | T '/' F | F;
F: number | '(' E ')';

@ignore [ \r\n\t]*
@match number \d+
'''

parser = LIBLR.create_parser(grammar)
print(parser('1+2*3'))


