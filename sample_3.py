import LIBLR

grammar = r'''
%token number

E: E '+' T          {add}
 | E '-' T          {sub}
 | T                {get1}
 ;

T: T '*' F          {mul}
 | T '/' F          {div}
 | F                {get1}
 ;

F: number           {getint}
 | '(' E ')'        {get2}
 ;

@ignore [ \r\n\t]*
@match number \d+
'''

class SemanticAction:
    def add (self, rule, args):
        return args[1] + args[3]
    def sub (self, rule, args):
        return args[1] - args[3]
    def mul (self, rule, args):
        return args[1] * args[3]
    def div (self, rule, args):
        return args[1] / args[3]
    def get1 (self, rule, args):
        return args[1]
    def get2 (self, rule, args):
        return args[2]
    def getint (self, rule, args):
        return int(args[1])

parser = LIBLR.create_parser(grammar, SemanticAction())
print(parser('1+2*3'))



