#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

download include files to build custom op
"""
import os
import ssl
import socket
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
socket.setdefaulttimeout(10)
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
BASE_URL = "https://raw.githubusercontent.com/microsoft/onnxruntime"
SECOND_URL = "include/onnxruntime/core/session"
SUPPORT_ORT_VERSION = ['1.5.2', '1.6.0', '1.8.0', '1.9.0', '1.16.0']


def get_ort_version():
    """ get the installed onnxruntime/onnxruntime-gpu version"""
    ort_version = ''
    try:
        import onnxruntime as ort
    except ImportError:
        pass
    else:
        ort_version = ort.__version__

    if ort_version == '':
        raise RuntimeError(
            "onnxruntime or onnxruntime-gpu is not installed, need to pip "
            "install onnxruntime or onnxruntime-gpu first. "
            "onnxruntime 1.9.0 or onnxruntime-gpu 1.9.0 is suggested.")
    if ort_version not in SUPPORT_ORT_VERSION:
        raise RuntimeError(
            "amct_onnx now only support onnxruntime/onnxruntime-gpu 1.6.0, "
            "1.8.0 or 1.9.0. if need to use opset 13, "
            "onnxruntime/onnxruntime-gpu 1.6.0 is required; "
            "opset 14 requires onnxruntime/onnxruntime-gpu 1.8.0 or 1.9.0")
    return "v{}".format(ort_version)


def get_file_urls(ort_version):
    """ get the header files urls according to the ort_version"""
    file_urls = {
        'onnxruntime_cxx_api.h':
            '{}/{}/{}/onnxruntime_cxx_api.h'.format(
                BASE_URL, ort_version, SECOND_URL),
        'onnxruntime_cxx_inline.h':
            '{}/{}/{}/onnxruntime_cxx_inline.h'.format(
                BASE_URL, ort_version, SECOND_URL),
        'onnxruntime_c_api.h':
            '{}/{}/{}/onnxruntime_c_api.h'.format(
                BASE_URL, ort_version, SECOND_URL),
        'onnxruntime_session_options_config_keys.h':
            '{}/{}/{}/onnxruntime_session_options_config_keys.h'.format(
                BASE_URL, ort_version, SECOND_URL),
    }
    if ort_version == 'v1.16.0':
        file_urls['onnxruntime_float16.h'] = '{}/{}/{}/onnxruntime_float16.h'.format(
            BASE_URL, ort_version, SECOND_URL)
    return file_urls


def download_inc_files():
    """download include files"""
    ort_version = get_ort_version()
    file_urls = get_file_urls(ort_version)
    for file_name, url in file_urls.items():
        inc_file = os.path.join(CUR_DIR, file_name)
        if os.path.exists(inc_file):
            print("[INFO]'{}' already exist, no need to download.".format(
                inc_file))
            continue
        urllib.request.urlretrieve(url, inc_file)
        print('[INFO]Download %s success.' % (inc_file))
