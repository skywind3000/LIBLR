# Preface

类似 Yacc/Bison 的 Parser Generator：

- 给定文法 BNF 生成 parser。
- 支持 LR(1) 和 LALR 两种算法。
- 支持 L 型 SDT 的语义动作。
- 支持对接自定义非规则的 Lexer。
- 基于优先级的冲突处理，类似 Yacc/Bison 的优先级体系。
- 实时解析，支持根据语义分析结果指导接下来的词法/语法分析。
- 通过 C11 的文法测试。
- 单文件实现，没有额外依赖，方便集成。

## Quick Start

### 简单例子

文法格式类似 Yacc/Bison，所有词法元素使用 `@ignore` 和 `@match` 定义。

```python
import LIBLR

# 注意这里是 r 字符串，方便后面写正则
# 所有词法规则用 @ 开头，从上到下依次匹配
grammar = r'''
start: WORD ',' WORD '!';

@ignore [ \r\n\t]*
@match WORD \w+
'''

parser = LIBLR.create_parser(grammar)
print(parser('Hello, World !'))
```

输出：

```
Node(Symbol('start'), ['Hello', ',', 'World', '!'])
```

默认没有加 Semantic Action 的话，会返回一颗带注释的语法分析树（annotated parse-tree）。

### 语义动作

语义动作（Semantic Action）是在生成式中用 `{name}` 表达的部分，对应 name 的方法会在回调中被调用：


```python
import LIBLR

# 注意这里是 r 字符串，方便后面写正则
grammar = r'''
# 事先声明终结符
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

# 忽略空白
@ignore [ \r\n\t]*
# 词法规则
@match number \d+
'''

# 定义语义动作：各个动作由类成员实现，每个方法的
# 第一个参数 rule 是对应的生成式
# 第二个参数 args 是各个部分的值，类似 yacc/bison 中的 $0-$N 
# args[1] 是生成式右边第一个符号的值，以此类推
# args[0] 是继承属性
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
```

输出结果

```
7
```

### 冲突处理

支持书写二义性规则，使用类似 Yacc/Bison 的优先级机制来确定发生冲突时使用哪个规则：

```python
import LIBLR

# 注意这里是 r 字符串，方便后面写正则
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
    # 注意，这里对应生成式 expr: '-' expr，因为前面有减号了
    # 所以右边 expr 的值对应的是 args[2]
    def negative (self, rule, args):
        return -(args[2])

parser = LIBLR.create_parser(grammar, SemanticAction())
print(parser('1+2*3+(5-2)*2'))
```

使用：`%left`, `%right`, `%noassoc` 同时定义优先级和结合方向，写在后面的终结符优先级更高。

默认一个生成式的优先级由最右边的终结符优先级决定，也可以显式的用 `%prec` 指明。

对比前面的例子，使用二意文法，能让 BNF 的书写精简不少，该程序的输出：

```
13
```

## Manual

### 语法规则

```
<头部> : <符号1> <符号2> <符号3> ';'
```

例子，同一头部的不同生成式可以展开写：

```
start: symbol1 symbol2;
start: symbol3 symbol4;
```

或者也可以用竖线分割合起来写：

```
start: symbol1 symbol2 | symbol3 symbol4 ;
```

看起来更精简些

### 内嵌字符串终结符

```
if_statement: 'if' condition 'then' result ;
```

可以直接用内嵌字符串表示终结符，无需额外定义词法规则。

### 终结符定义

非内嵌字符串的终结符和 YACC 一样需要事先定义一下：

```
%token NUMBER 
```

然后后面用 `@match NAME pattern` 来定义正则：

```
@match NUMBER \d+
```

词法规则从上到下的顺序进行匹配，除了 `@match` 外，还可以用 `@ignore` 来规定忽略哪些东西，比如你想跳过空格时，你可以在最开头写：

```
@ignore [ \r\n\t]*
```

否则碰到空格就 unexpected character 掉了。

### 语义动作

语义动作用 `{name}` 的形式在文法中声明，可以位于生成式右边的任意位置：

```
start: {action1} symbol1 {action2} symbol2 {acion3} ;
```

在 create_parser 的时候，第二个参数传进去一个回调对象：

```python
callback = SemanticAction()
parser = LIBLR.create_parser(grammar, callback)
```

当动作发生时，callback 对象内部的同名方法会被调用，格式：

```python
def my_action(rule, args): 
```

第一个参数是对应生成式，第二个参数是生成式各个符号的值，其中 args[1] 是第一个符号的值，args[2] 是第二个符号的值，同 Yacc 一样，args[0] 表示继承属性。

语义动作计算好以后，使用 return 返回它的值。之所以不像 Yacc 那样直接把 C 代码写道文法动作里，只写一个名字，再靠外部的 callback 具体实现，就是为了不和某种语言耦合太深，将来可以支持导出其它语言。

## Samples

| 文件名 | 说明 |
|-|-|
| [sample_1.py](sample_1.py) | Hello, World |
| [sample_2.py](sample_2.py) | 表达式语法分析树 |
| [sample_3.py](sample_3.py) | 表达式计算 |
| [sample_4.py](sample_4.py) | 二义文法演示 |
| [sample_json.py](sample_json.py) | 带注释的 json 解析 |

## TODO 

- [ ] Better error messages.
- [ ] Translate EBNF to BNF.
- [ ] Generate parsers in more languages.
- [ ] Implement GLR.

