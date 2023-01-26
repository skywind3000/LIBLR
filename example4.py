import LIBLR

grammar = r'''
%token NUMBER

%left '+' '-'
%left '*' '/' '%'
%right UMINUS

expr: expr '+' expr             {add}
    | expr '-' expr             {sub}
    | expr '*' expr             {mul}
    | expr '/' expr             {div}
    | '(' expr ')'              {get2}
    | '-' expr %prec UMINUS     {negative}
    | NUMBER                    {getint}
    ;

@ignore [ \r\n\t]*
@match NUMBER \d+
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
    def negative (self, rule, args):
        return -(args[2])

parser = LIBLR.create_parser(grammar, SemanticAction())
print(parser('1+2*3+(5-2)*2'))




