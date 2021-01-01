# -*- coding: utf-8 -*-

#
# OS related library functions
#

import re, os, subprocess, json

def run_command(str_array):
    x = subprocess.Popen(str_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = x.stderr.read()
    if len(err) > 1:
        print(err.decode("utf8"))
    return x.stdout.read()