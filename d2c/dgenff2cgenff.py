#!/usr/bin/python

import sys
import re

def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

adict = eval(open(sys.argv[1]).read())
print(adict)
with open(sys.argv[2]) as text:
   new_text = multiple_replace(adict, text.read())
   print(new_text)
with open(sys.argv[2]+".new", "w") as result:
   result.write(new_text)
