import sys
import pprint
import LIBLR

from LIBLR import cstring

class JsonAction:

    def get1 (self, rule, args):
        return args[1]

    def get_string (self, rule, args):
        return cstring.string_unquote(args[1])

    def get_number (self, rule, args):
        text = args[1]
        v = float(text)
        if v.is_integer():
            return int(v)
        return v

    def get_true (self, rule, args):
        return True

    def get_false (self, rule, args):
        return False

    def get_null (self, rule, args):
        return None

    def list_empty (self, rule, args):
        return []

    def list_one (self, rule, args):
        return [args[1]]

    def list_many (self, rule, args):
        return args[1] + [args[3]]

    def get_array (self, rule, args):
        return args[2]

    def item_pair (self, rule, args):
        name = cstring.string_unquote(args[1])
        value = args[3]
        return (name, value)

    def get_object (self, rule, args):
        obj = {}
        for k, v in args[2]:
            obj[k] = v
        return obj

parser = LIBLR.create_parser_from_file('grammar/json.txt',
                                       JsonAction(),
                                       algorithm = 'lalr')

text = open('sample/launch.json').read()

pprint.pprint(parser(text))



