import sys


def progress(i, n, prestr=""):
    sys.stdout.write("\r{}: {}\{}".format(prestr, i, n))
