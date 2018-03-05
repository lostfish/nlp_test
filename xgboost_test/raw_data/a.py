import sys
for line in sys.stdin:
    s = line.rstrip('\n').split('\t')
    a = [""]*26
    a[0] = s[0]
    a[25] = s[25]
    print '\t'.join(a)
