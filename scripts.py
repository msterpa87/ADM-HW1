# Say "Hello, World!" With Python
print("Hello, World!")

# Write a function
def is_leap(year):
    leap = False
    
    leap = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    
    return leap

year = int(input())
print(is_leap(year))

# Print Function
if __name__ == '__main__':
    n = int(input())
    print("".join(list(map(str,range(1,n+1)))))


# Python If-Else
if __name__ == '__main__':
    n = int(raw_input().strip())
    if n % 2 == 1:
        print("Weird")
    else:
        if 2 <= n <= 5 or n > 20:
            print("Not Weird")
        else:
            print("Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())

# Loops
if __name__ == '__main__':
    n = int(raw_input())

    for i in range(n):
        print(i**2)

# Lists
if __name__ == '__main__':
    N = int(input())
    L = []

    for _ in range(N):
        cmd = input().split()
        
        if cmd[0] == "pop":
            L.pop()
        elif cmd[0] == "sort":
            L.sort()
        elif cmd[0] == "reverse":
            L.reverse()
        elif cmd[0] == "print":
            print(L)
        elif cmd[0] == "insert":
            L.insert(int(cmd[1]), int(cmd[2]))
        elif cmd[0] == "remove":
            L.remove(int(cmd[1]))
        elif cmd[0] == "append":
            L.append(int(cmd[1]))

# List Comprehension
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())

    l = [[x,y,z] for x in range(x+1) for y in range(y+1) for z in range(z+1) if x+y+z != n]
    print(l)


# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())

    winner = max(arr)
    arr_wo_winner = [x for x in arr if x != winner]
    print(max(arr_wo_winner))


# Nested Lists
names = []
scores = []

for _ in range(int(raw_input())):
    name = raw_input()
    score = float(raw_input())
    names.append(name)
    scores.append(score)

min_score = min(scores)
l = [[x,y] for x,y in sorted(zip(scores,names)) if x != min_score]
new_min = l[0][0]
l = list(filter(lambda x: x[0] == new_min, l))
for x,y in l:
    print(y)


# Finding the Percentage
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    
    scores = student_marks[query_name]
    avg = sum(scores)/len(scores)
    print("{:.2f}".format(avg))

# Tuples
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    print(hash(tuple(integer_list)))

# Find a string
def count_substring(string, sub_string):
    count = 0
    n = len(string)
    m = len(sub_string)

    for i in range(n):
        if string[i] == sub_string[0]:
            found = True
            for j in range(1,m):
                if i+j >= n or string[i+j] != sub_string[j]:
                    found = False
                    break
            if found:
                count += 1

    return count

if __name__ == '__main__':

# sWAP cASE
def swap_case(s):
    string = ""

    for c in s:
        if c.isupper():
            string += c.lower()
        else:
            string += c.upper()

    return string

# String Split and Join
def split_and_join(line):
    return "-".join(line.split())

# What's Your Name?
def print_full_name(a, b):
    print "Hello {} {}! You just delved into python.".format(a,b)

# Mutations
def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    return "".join(l)

# String Validators
if __name__ == '__main__':
    s = raw_input()
    digit = False
    alnum = False
    alpha = False
    lower = False
    upper = False

    for c in s:
        if c.isalnum():
            alnum = True
        if c.isdigit():
            digit = True
        if c.isalpha():
            alpha = True
        if c.islower():
            lower = True
        if c.isupper():
            upper = True
    
    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)

# Text Alignment

thickness = int(raw_input())
c = 'H'

for i in range(thickness):
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

for i in range((thickness+1)/2):
    print (c*thickness*5).center(thickness*6)    

for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)    

for i in range(thickness):
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)

# Text Wrap
def wrap(string, max_width):
    wrapper = textwrap.TextWrapper(width=max_width)
    return "\n".join(wrapper.wrap(string))

# Designer Door Mat
N, M = map(int, raw_input().split())
rows = (N-1)/2
for i in range(rows):
    print((".|."*(1+2*i)).center(M,"-"))
print("WELCOME".center(M,"-"))
for i in reversed(range(rows)):
    print((".|."*(1+2*i)).center(M,"-"))

# String Formatting
def print_formatted(number):

    width = len(bin(number)[2:])

    for i in range(1,number+1):
        x = str(i).rjust(width)
        y = oct(i)[1:].rjust(width)
        w = hex(i)[2:].upper().rjust(width)
        z = bin(i)[2:].rjust(width)
        print("{} {} {} {}".format(x,y,w,z))

# Alphabet Rangoli
import string

def print_rangoli(size):
    width = (size-1)*4+1
    L = list(string.ascii_lowercase)[:size]
    rows = []

    for row in range(1,size+1):
        L1 = L[size-row:]
        L2 = L[size-row+1:size]
        L3 = list(reversed(L1))+L2
        rows.append("-".join(L3))

    rows += list(reversed(rows[:-1]))
    for row in rows:
        print(row.center(width,"-"))

# Capitalize!
def solve(s):
    s = list(s)
    s[0] = s[0].upper()
    
    for i,c in enumerate(s):
        if i > 0 and s[i-1] == ' ':
            s[i] = c.upper()
    
    return "".join(s)

# The Minion Game
def minion_game(string):
    stuart = 0
    kevin = 0
    n = len(string)
    vowels = ['A','E','I','O','U']

    for i in range(n):
        if string[i] in vowels:
            kevin += n-i
        else:
            stuart += n-i

    if kevin == stuart:
        print("Draw")
    elif kevin > stuart:
        print("Kevin {}".format(kevin))
    else:
        print("Stuart {}".format(stuart))

# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(len(string)/k):
        s = list(string[k*i:k*(i+1)])
        seen = {}
        u = ""

        for i,x in enumerate(s):
            if x not in seen:
                seen[x] = True
                u += x
        print(u)

# Introduction to Sets
def average(array):
    s = set(array)
    return sum(s)/len(s)

# No Idea!
n, m = map(int, raw_input().split())
array = map(int, raw_input().split())
A = set(map(int, raw_input().split()))
B = set(map(int, raw_input().split()))
h = 0
occurrence = {}
for x in array:
    if x not in occurrence:
        occurrence[x] = 1
    else:
        occurrence[x] += 1

A1 = A.intersection(set(array))
B1 = B.intersection(set(array))

for x in A1:
    h += occurrence[x]
for x in B1:
    h -= occurrence[x]

print(h)

# Symmetric Difference
M = int(raw_input())
set1 = set(map(int, raw_input().split()))
N = int(raw_input())
set2 = set(map(int, raw_input().split()))
D = set1.difference(set2).union(set2.difference(set1))
D = sorted(list(D))
for x in D:
    print(x)

# Set .add()
n = int(raw_input())
s = set()

for i in range(n):
    s.add(raw_input())

print(len(s))

# Set .discard(), .remove() & .pop()
n = input()
s = set(map(int, raw_input().split()))
N = int(raw_input())

for i in range(N):
    op = raw_input()
    if op[0] == "p":
        s.pop()
    else:
        op, x = op.split()
        x = int(x)

        if op[0] == "r":
            s.remove(x)
        elif op[0] == "d":
            s.discard(x)

print(sum(s))

# Set .union() Operation
n = raw_input()
students = set(map(int, raw_input().split()))
n = raw_input()
students = students.union(set(map(int, raw_input().split())))
print(len(students))

# Set .intersection() Operation
n = raw_input()
S = set(map(int,raw_input().split()))
n = raw_input()
S = S.intersection(set(map(int, raw_input().split())))
print(len(S))

# Set .difference() Operation
raw_input()
S = set(map(int, raw_input().split()))
raw_input()
S = S.difference(set(map(int, raw_input().split())))
print(len(S))

# Set .symmetric_difference() Operation
raw_input()
S1 = set(map(int, raw_input().split()))
raw_input()
S2 = set(map(int, raw_input().split()))
S = S1.difference(S2).union(S2.difference(S1))
print(len(S))

# Set Mutations
a = int(raw_input())
A = set(map(int, raw_input().split()))
n = int(raw_input())
for i in range(n):
    op, size = raw_input().split()
    size = int(size)
    S = set(map(int, raw_input().split()))
    if op == "intersection_update":
        A.intersection_update(S)
    elif op == "update":
        A.update(S)
    elif op == "symmetric_difference_update":
        A.symmetric_difference_update(S)
    elif op == "difference_update":
        A.difference_update(S)

print(sum(A))

# The Captain's Room
n = int(raw_input())
L = list(map(int, raw_input().split()))
S = set(L)

for x in S:
    L.remove(x)

print(S.difference(set(L)).pop())

# Check Subset
n = int(raw_input())
for i in range(n):
    a = int(raw_input())
    A = set(map(int, raw_input().split()))
    b = int(raw_input())
    B = set(map(int, raw_input().split()))

    I = B.intersection(A)
    if I == A:
        print("True")
    else:
        print("False")

# Check Strict Superset
A = set(map(int, raw_input().split()))
n = int(raw_input())
flag = True

for i in range(n):
    S = set(map(int, raw_input().split()))
    if len(S.difference(A)) > 0:
        flag = False
        break

print(flag)

# collections.Counter()
from collections import Counter

n = int(raw_input())
shoes = list(map(int, raw_input().split()))
n_customers = int(raw_input())
counter = Counter(shoes)
earned = 0

for i in range(n_customers):
    size, pay = list(map(int, raw_input().split()))
    if size in counter and counter[size] > 0:
        counter[size] -= 1
        earned += pay

print(earned)

# DefaultDict Tutorial
from collections import defaultdict

n, m = list(map(int, raw_input().split()))
A = defaultdict(list)

for i in range(n):
    x = raw_input()
    A[x].append(i+1)

for i in range(m):
    x = raw_input()
    if x in A:
        print(" ".join(map(str, A[x])))
    else:
        print("-1")

# Collections.namedtuple()
from collections import namedtuple

n = int(raw_input())
Student = namedtuple('Student',raw_input().split())
students = []
marks = 0.0

for i in range(n):
    marks += int(Student._make(raw_input().split()).MARKS)

print(round(marks/n,2))

# Collections.OrderedDict()
from collections import OrderedDict

d = OrderedDict()
n = int(raw_input())
for i in range(n):
    row_parts = raw_input().split()
    price = int(row_parts[-1])
    item = " ".join(row_parts[:-1])

    if item not in d:
        d[item] = price
    else:
        d[item] += price

for key in d.keys():
    print("{} {}".format(key, d[key]))

# Word Order
from collections import defaultdict

n = int(raw_input())
d = defaultdict(int)
l = list()

for i in range(n):
    w = raw_input()
    d[w] += 1
    if d[w] == 1:
        l.append(w)

print(len(d.keys()))
print(" ".join(map(lambda w: str(d[w]), l)))

# Collections.dequeu()
from collections import deque

d = deque()
n = int(raw_input())

for i in range(n):
    op = raw_input()
    if op[:3] != "pop":
        op, x = op.split()
        x = int(x)
    if op == "append":
        d.append(x)
    elif op == "appendleft":
        d.appendleft(x)
    elif op == "pop":
        d.pop()
    elif op == "popleft":
        d.popleft()

print(" ".join(map(str, d)))

# Company Logo
import math
import os
import random
import re
import sys
from collections import defaultdict


if __name__ == '__main__':
    s = raw_input()
    d = defaultdict(int)
    d_sort = defaultdict(list)

    for c in list(s):
        d[c] += 1
       
    for key,val in d.items():
        d_sort[val].append(key)
    
    i = 0

    for key in reversed(sorted(d_sort.keys())):
        for val in sorted(d_sort[key]):
            print("{} {}".format(val,key))
            i += 1
            if i == 3:
                break
        if i == 3:
            break

# Piling Up!
from collections import deque

T = int(raw_input())

for i in range(T):
    n = int(raw_input())
    d = deque(map(int, raw_input().split()))
    last = d.popleft()
    result = "Yes"

    try:
        x = d.pop()
        if x > last:
            d.appendleft(last)
            last = x
    except:
        print("Yes")
        continue

    while True:
        try:
            x = d.popleft()
        except:
            break

        if x > last:
            try:
                d.appendleft(x)
                x = d.pop()
                if x > last:
                    result = "No"
                    break
                else:
                    last = x
            except:
                break
        else:
            try:
                y = d.pop()
                if y > x and y <= last:
                    d.appendleft(x)
                    last = y
                else:
                    d.append(y)
            except:
                break
    
    print(result)

# Calendar Module
import calendar

days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
month, day, year = list(map(int, raw_input().split()))
print(days[calendar.weekday(year, month, day)])

# Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    dt1 = datetime.strptime(t1, "%a %d %b %Y %X %z")
    dt2 = datetime.strptime(t2, "%a %d %b %Y %X %z")
    ms = (dt1 - dt2).total_seconds()
    return int(abs(ms))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(str(delta) + '\n')

    fptr.close()

# Exceptions
for _ in range(int(input())):
    try:
        a,b = map(int, input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

# Zipped
_, n = list(map(int, input().split()))
scores = [list(map(float, input().split())) for _ in range(n)]

for x in list(zip(*scores)):
    v = round(sum(x)/n, 1)
    print(v)

# Athlete Sort
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

    for x in sorted(arr, key=lambda x: x[k]):
        print(" ".join(map(str, x)))

# ginortS
s = input()
lower = sorted(list(filter(lambda x: x.islower(), s)))
upper = sorted(list(filter(lambda x: x.isupper(), s)))
odd = sorted(list(filter(lambda x: x.isdigit() and int(x) % 2 == 1, s)))
even = sorted(list(filter(lambda x: x.isdigit() and int(x) % 2 == 0, s)))

s = lower + upper + odd + even
print("".join(s))

# Map and Lambda Functions
cube = lambda x: x**3 complete the lambda function 

def fibonacci(n):
    return a list of fibonacci numbers
    if n == 0:
        return []
    elif n == 1:
        return [0]

    F = [0,1]

    for i in range(2,n):
        F.append(F[i-2] + F[i-1])
    
    return F

# Detect Floating Point Number
import re


T = int(input())

for _ in range(T):
    s = input()
    if re.search(r"^[\+|-]?\d*\.\d+$", s) == None:
        print("False")
    else:
        print("True")

# Re.split()
regex_pattern = r",|\."

# Group(), Groups() & Groupdict()
import re

m = re.search(r'([0-9|a-z|A-Z])\1', input())
try:
    print(m.group(1))
except:
    print("-1")

# Re.findall() & Re.finditer()
import re

V = "aeiouAEIOU"
C = "[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]"
m =  re.findall("(?="+C+"(["+V+"]{2,})"+C+")", input())
if len(m) == 0:
    print("-1")
else:
    for s in m:
        print(s)

# Re.start() & Re.end()
import re
S = input()
k = input()
matches = list(re.finditer("(?=(%s))" % k, S))
if matches == []:
    print("(-1, -1)")

for m in matches:
    print("({}, {})".format(m.start(1),m.end(1)-1))

# Regex Substitution
import re

def swap(m):
    s = m.group(0)
    if s == "&&":
        return "and"
    elif s == "||":
        return "or"

N = int(input())
for _ in range(N):
    s = input()
    print(re.sub(r"(?<= )(&&|\|\|)(?= )",swap,s))

# Validating phone numbers
import re
for _ in range(int(input())):
    if re.search(r"^[789][0-9]{9}$", input()) == None:
        print("NO")
    else:
        print("YES")

# Validating and Parsing Email Adresses
import email.utils
import re

for _ in range(int(input())):
    addr = email.utils.parseaddr(input())
    if re.search(r"^[a-zA-Z][\w|.|-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$", addr[1]) != None:
        print(email.utils.formataddr(addr))

# HTML Parser - Part 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for name, value in attrs:
            print("-> {} > {}".format(name, value))
            
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for name, value in attrs:
            print("-> {} > {}".format(name, value))

    def handle_endtag(self, tag):
        print("End   :", tag)

    def handle_commenct(self, data):
        pass

parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data.strip())
  
    def handle_comment(self, data):
        if "\n" in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.strip())
  

html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for name,value in attrs:
            print("-> {} > {}".format(name,value))

parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

# Validating UID
import re

def check_uid(s):
    if re.search("[A-Z].*[A-Z]",s) == None:
        return "Invalid"
    if re.search("[0-9].*[0-9].*[0-9]",s) == None:
        return "Invalid"
    if re.search("^[a-zA-Z0-9]+$",s) == None:
        return "Invalid"
    if re.search(r"([a-zA-Z0-9]).*\1+",s) != None:
        return "Invalid"
    if len(s) != 10:
        return "Invalid"
    return "Valid"

T = int(input())
for _ in range(T):
    print(check_uid(input()))

# Validating Credit Card Numbers
import re

for _ in range(int(input())):
    s = input()
    if re.match("[456][0-9]{3}-?[0-9]{4}-?[0-9]{4}-?[0-9]{4}$", s) != None:
        if re.search(r"([0-9])\1{3,}", s.replace("-","")) != None:
            print("Invalid")
        else:
            print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[^0]\d{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

# Matrix Script
import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

l = list("".join(matrix))
s = ""
for i in range(m):
    s += "".join(l[i::m])
s = re.sub(r"([0-9a-zA-Z])[^0-9a-zA-Z]+([0-9a-zA-Z])",r"\1 \2",s)
print(s)

# XML 1 - Find the Score
def get_attr_number(node):
    your code goes here
    score = 0

    for elem in node.iter():
        score += len(elem.attrib)
    
    return score

# XML 2 - Find the Maximum Depth
maxdepth = -1
def depth(elem, level):
    global maxdepth
    your code goes here
    if level == maxdepth:
        maxdepth += 1
    
    for child in elem:
        depth(child, level+1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        complete the function
        for i,x in enumerate(l):
            x = x[len(x)-10:]
            l[i] = "+91 " + x[:5] + " " + x[5:]
        l.sort()
        return f(l)
    return fun

# Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        complete the function
        people = list(map(lambda x: [x[0],x[1],int(x[2]), x[3]], people))
        return list(map(f, sorted(people, key=operator.itemgetter(2))))
    return inner

# Arrays
def arrays(arr):
    return numpy.flip(numpy.array(arr, dtype=numpy.float32))

# Shape and Reshape
import numpy

arr = list(map(int, input().split()))
print(numpy.array(arr).reshape((3,3)))

# Transpose and Flatten
import numpy

arr = []
N, M = list(map(int, input().split()))
for _ in range(N):
    arr.append(list(map(int, input().split())))

A = numpy.array(arr).reshape(N,M)
print(numpy.transpose(A))
print(A.flatten())

# Concatenate
import numpy

N, M, P = list(map(int, input().split()))
arr1 = []
arr2 = []
for _ in range(N):
    arr1.append(list(map(int, input().split())))
for _ in range(M):
    arr2.append(list(map(int, input().split())))
arr1 = numpy.array(arr1).reshape(N,P)
arr2 = numpy.array(arr2).reshape(M,P)
print(numpy.concatenate((arr1, arr2), axis=0))

# Zeros and Ones
import numpy

dim = list(map(int, input().split()))
print(numpy.zeros(dim, dtype=numpy.int))
print(numpy.ones(dim, dtype=numpy.int))

# Eye and Identity
import numpy

N,M = list(map(int, input().split()))
numpy.set_printoptions(sign=' ')
print(numpy.eye(N,M))

# Array Mathematics
import numpy

N,M = list(map(int, input().split()))
arr1 = []
arr2 = []
for _ in range(N):
    arr1.append(list(map(int, input().split())))
for _ in range(N):
    arr2.append(list(map(int, input().split())))
A = numpy.array(arr1)
B = numpy.array(arr2)
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

# Floor, Ceil and Rint
import numpy

numpy.set_printoptions(sign=' ')

A = numpy.array(list(map(float, input().split())))
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Sum and Prod
import numpy

N, _ = list(map(int,input().split()))
arr = []
for _ in range(N):
    arr.append(list(map(int,input().split())))
A = numpy.array(arr)
print(numpy.prod(numpy.sum(A,axis=0)))

# Min and Max
import numpy

N = int(input().split()[0])
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))

A = numpy.array(A)
print(numpy.max(numpy.min(A,axis=1)))

# Mean, Var and Std
import numpy

N = int(input().split()[0])
numpy.set_printoptions(legacy='1.13')
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))
A = numpy.array(A)
print(numpy.mean(A,axis=1))
print(numpy.var(A,axis=0))
print(numpy.std(A,axis=None))

# Dot and Cross
import numpy

N = int(input())
arr1 = []
arr2 = []
for _ in range(N):
    arr1.append(input().split())
for _ in range(N):
    arr2.append(input().split())
A = numpy.array(arr1, int)
B = numpy.array(arr2, int)
print(numpy.dot(A,B))

# Inner and Outer
import numpy

A = numpy.array(input().split(),int)
B = numpy.array(input().split(),int)
print(numpy.inner(A,B))
print(numpy.outer(A,B))

# Polynomials
import numpy

print(numpy.polyval(numpy.array(input().split(),float),float(input())))
# Linear Algebra
import numpy

print(round(numpy.linalg.det(numpy.array([input().split() for _ in range(int(input()))], float)),2))

# Birthday Cake Candles
def birthdayCakeCandles(candles):
    max_candle = max(candles)
    return candles.count(max_candle)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    try:
        n = (x1-x2)/(v2-v1)
    except:
        return "NO"
   print(n)
    if n - int(n) == 0 and n>0:
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral advertising
def viralAdvertising(n):
    total = like = 2
    for _ in range(1,n):
        like = math.floor(like*3/2)
        total += like
    return total

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum
def superDigit(n, k):
    p = list(str(n))
    while(len(p) > 1):
        p = list(str(sum(map(int,list(p)))))
    return sum(list(map(int,str(int(p[0])*k))))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion sort 1
def insertionSort1(n, arr):
    x = arr[-1]
    for i in range(n-2,-1,-1):
        if arr[i] > x:
            arr[i+1] = arr[i]
        elif arr[i] <= x:
            arr[i+1] = x
            print(" ".join(map(str,arr)))
            break
        print(" ".join(map(str,arr)))
        if i == 0:
            arr[i] = x
            print(" ".join(map(str,arr)))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion sort 2
def insertionSort2(n, arr):
    for i in range(1,len(arr)):
        if arr[i] > arr[i-1]:
            print(" ".join(map(str,arr)))
        else:
            j = i
            while j>=1 and arr[j] < arr[j-1]:
                y = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = y
                j -= 1
            print(" ".join(map(str,arr)))


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)