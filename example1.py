import LIBLR

grammar = r'''
start: WORD ',' WORD '!';

@ignore [ \r\n\t]*
@match WORD \w+
'''

parser = LIBLR.create_parser(grammar)
print(parser('Hello, World !'))

