# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
from enum import Enum
from time import time
import logging


class Device(Enum):
    CUDA = 1
    IPU = 2
    X86 = 3

ODLA_DYNAMIC_BATCH = 0
ODLA_MIN_BATCH_SIZE = 1
ODLA_MAX_BATCH_SIZE = 2
ODLA_OPT_BATCH_SIZE = 3
ODLA_RUN_BATCH_SIZE = 4
ODLA_DYNAMIC_SHAPE = 5
ODLA_MIN_SHAPE = 6
ODLA_MAX_SHAPE = 7
ODLA_OPT_SHAPE = 8
ODLA_BF16_MODE = 9
ODLA_FP16_MODE = 10
ODLA_USE_SIM_MODE = 11
ODLA_PROCESSOR_NUM = 12
ODLA_BATCHES_PER_STEP = 13
ODLA_USE_DATA_TYPE = 14
ODLA_LOAD_ENGINE_MODE = 15
ODLA_USE_DLA_CORE = 16
ODLA_QUANT_TABLE = 17
ODLA_QUANT_TABLE_SIZE = 18
ODLA_ENABLE_ENGINE_CACHE = 19
ODLA_CACHE_DIR = 20

class ValueShape(Structure):
    _fields_ = [("size", c_int32), ("dims", c_int64 * 10)]


class ValueType(Structure):
    _fields_ = [("element_type", c_int32), ("shape", ValueShape)]


class ODLAModel:
    def __init__(self, so_file):
        self.logger = logging.getLogger(__name__)
        self.so_file = so_file
        self.h = None
        self.buffers = []

    def __del__(self):
        if self.h:
            if self.ctx:
                self.h.odla_DestroyContext(self.ctx)
            if self.comp:
                self.h.odla_DestroyComputation(self.comp)

    def Load(self):
        if self.h is None:
            self.h = CDLL(self.so_file)
        self.comp = c_void_p(0)
        self.h.odla_CreateComputation(pointer(self.comp))

        # # TODO:
        # use_sim = c_bool(True)
        # self.h.odla_SetComputationItem(self.comp, 7, pointer(use_sim))    # 7 ??? mry

        self.h.model_helper(self.comp)
        n = c_int32(-1)
        self.h.odla_GetNumOfArgsFromComputation(self.comp, pointer(n))
        self.nr_args = n.value

        nr_args = c_int32(-1)
        self.h.odla_GetNumOfOutputsFromComputation(self.comp, pointer(n))
        self.nr_outputs = n.value











        # self.in_vals = []
        # for idx in range(0, self.nr_args):
        #     arg_v = c_void_p(0)
        #     self.h.odla_GetArgFromComputationByIdx(self.comp, idx, pointer(arg_v))
        #     vt = ValueType()
        #     self.h.odla_GetValueType(arg_v, pointer(vt))
        #     self.in_vals.append((arg_v.value, vt))

            # mry todo bindargs



        # if not is_dynamic_shape
        # self.out_vals = []
        # for idx in range(0, self.nr_outputs):
        #     out = c_void_p(0)
        #     self.h.odla_GetOutputFromComputationByIdx(self.comp, idx, pointer(out))
        #     vt = ValueType()
        #     self.h.odla_GetValueType(out, pointer(vt))
        #     n = 1
        #     for r in range(0, vt.shape.size):
        #         n *= vt.shape.dims[r]
        #     self.out_vals.append((out, vt, n))
        #     buf = (c_float * n)() # FIXME: handle types
        #     self.h.odla_BindToOutput(out, buf, self.ctx)
        #     self.buffers.append(buf)

    




    def Execute(self, data):

        # mry if not is_dynamic_shape ?? load的时候传入data？

        # mry if is_dynamic_shape
        input0_idx = 0
        input0_v = c_void_p(0)
        is_dynamic_shape = c_bool(True)
        dims_array = c_int64 * 10
        input0_min_shape = ValueShape(c_int32(4), dims_array(1, 3, 1, 1))
        input0_max_shape = ValueShape(c_int32(4), dims_array(1, 3, 1000, 2000))
        input0_opt_shape = ValueShape(c_int32(4), dims_array(1, 3, 960, 1280))

        self.h.odla_GetArgFromComputationByIdx(self.comp, input0_idx, pointer(input0_v))
        self.h.odla_SetComputationItem(self.comp, ODLA_DYNAMIC_SHAPE, pointer(is_dynamic_shape))
        self.h.odla_SetValueShapeInfo(input0_v, ODLA_MIN_SHAPE, input0_min_shape)
        self.h.odla_SetValueShapeInfo(input0_v, ODLA_MAX_SHAPE, input0_max_shape)
        self.h.odla_SetValueShapeInfo(input0_v, ODLA_OPT_SHAPE, input0_opt_shape)

        self.ctx = c_void_p(0)
        self.h.odla_CreateContext(pointer(self.ctx))

        input0_real_shape = ValueShape(c_int32(4), dims_array(1, 3, 960, 1280))
        self.h.odla_SetRuntimeShape(self.ctx, input0_v, input0_real_shape)


        self.in_vals = []
        for idx in range(0, self.nr_args):
            arg_v = c_void_p(0)
            self.h.odla_GetArgFromComputationByIdx(self.comp, idx, pointer(arg_v))
            vt = ValueType()
            self.h.odla_GetValueType(arg_v, pointer(vt))
            self.in_vals.append((arg_v.value, vt))
        for idx, v in enumerate(self.in_vals):
            self.h.odla_BindToArgument(
                v[0], data[idx].ctypes.data_as(c_void_p), self.ctx
            )

        # else if  is_dynamic_shape
        self.out_vals = []
        for idx in range(0, self.nr_outputs):
            out = c_void_p(0)
            self.h.odla_GetOutputFromComputationByIdx(self.comp, idx, pointer(out))
            vs = ValueShape()
            self.h.odla_GetRuntimeShape(self.ctx, out, pointer(vs))
            n = 1
            for r in range(0, vs.size):
                n *= vs.dims[r]
            self.out_vals.append((out, vt, n)) #mry todo vt->vs
            buf = (c_float * n)() # FIXME: handle types
            self.h.odla_BindToOutput(out, buf, self.ctx)
            

            self.buffers.append(buf)


        s = time()
        self.h.odla_ExecuteComputation(self.comp, self.ctx, 0, c_void_p(0))
        t = time()
        self.logger.info("Execution time:" + str(t - s) + " sec(s)")

        return self.buffers
