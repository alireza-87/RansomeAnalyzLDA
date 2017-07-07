#!/usr/bin/env python

# This file is part of Androguard.
#
# Copyright (C) 2012, Anthony Desnos <desnos at t0t0.fr>
# All rights reserved.
#
# Androguard is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Androguard is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Androguard.  If not, see <http://www.gnu.org/licenses/>.

import sys, os
from optparse import OptionParser

from androguard.core.bytecodes import apk, dvm
from androguard.core.data import data
from androguard.core.analysis import analysis, ganalysis
from androguard.core import androconf

option_0 = {'name': ('-i', '--input'), 'help': 'filename input (dex, apk)', 'nargs': 1}
option_1 = {'name': ('-o', '--output'), 'help': 'directory output', 'nargs': 1}

options = [option_0, option_1]


def create_directory(class_name, output):
    output_name = output
    if output_name[-1] != "/":
        output_name = output_name + "/"

    try:
        os.makedirs(output_name + class_name)
    except OSError:
        pass


def create_directories(vm, output):
    for class_name in vm.get_classes_names():
        z = os.path.split(class_name)[0]
        create_directory(z[1:], output)


def create_graph(input,output):
    if input != None and output != None:

        ret_type = androconf.is_android(input)
        vm = None
        a = None
        if ret_type == "APK":
            a = apk.APK(input)
            if a.is_valid_APK():
                vm = dvm.DalvikVMFormat(a.get_dex())
            else:
                print "INVALID APK"
        elif ret_type == "DEX":
            try:
                vm = dvm.DalvikVMFormat(open(input, "rb").read())
            except Exception, e:
                print "INVALID DEX", e

        vmx = analysis.VMAnalysis(vm)
        gvmx = ganalysis.GVMAnalysis(vmx, a)

        create_directories(vm, output)

        dd = data.Data(vm, vmx, gvmx, a)

        buff = dd.export_methodcalls_to_gml()
        androconf.save_to_disk(buff, output + "/" + "methodcalls.graphml")