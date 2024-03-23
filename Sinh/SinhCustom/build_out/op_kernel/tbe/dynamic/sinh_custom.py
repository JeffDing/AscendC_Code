
import os, sys
import ctypes
import json
import shutil
from tbe.common.platform import get_soc_spec
from tbe.common.utils import para_check
from tbe.tikcpp import compile_op, replay_op, check_op_cap, generalize_op_params, get_code_channel, OpInfo
from tbe.common.buildcfg import get_default_build_config
from impl.util.platform_adapter import tbe_register
from tbe.common.buildcfg import get_current_build_config
PYF_PATH = os.path.dirname(os.path.realpath(__file__))

DTYPE_MAP = {"float32": ["DT_FLOAT", "float"],
    "float16": ["DT_FLOAT16", "half"],
    "int8": ["DT_INT8", "int8_t"],
    "int16": ["DT_INT16", "int16_t"],
    "int32": ["DT_INT32", "int32_t"],
    "int64": ["DT_INT64", "int64_t"],
    "uint1": ["DT_UINT1", "uint8_t"],
    "uint8": ["DT_UINT8", "uint8_t"],
    "uint16": ["DT_UINT16", "uint16_t"],
    "uint32": ["DT_UINT32", "uint32_t"],
    "uint64": ["DT_UINT64", "uint64_t"],
    "bool": ["DT_BOOL", "bool"],
    "double": ["DT_DOUBLE", "double"],
    "dual": ["DT_DUAL", "unknown"],
    "dual_sub_int8": ["DT_DUAL_SUB_INT8", "unknown"],
    "dual_sub_uint8": ["DT_DUAL_SUB_UINT8", "unknown"],
    "string": ["DT_STRING", "unknown"],
    "complex64": ["DT_COMPLEX64", "unknown"],
    "complex128": ["DT_COMPLEX128", "unknown"],
    "qint8": ["DT_QINT8", "unknown"],
    "qint16": ["DT_QINT16", "unknown"],
    "qint32": ["DT_QINT32", "unknown"],
    "quint8": ["DT_QUINT8", "unknown"],
    "quint16": ["DT_QUINT16", "unknown"],
    "resource": ["DT_RESOURCE", "unknown"],
    "string_ref": ["DT_STRING_REF", "unknown"],
    "int4": ["DT_INT4", "int8_t"],
    "bfloat16": ["DT_BF16", "bfloat16_t"]}

def get_dtype_fmt_options(__inputs__, __outputs__):
    options = []
    for x in __inputs__ + __outputs__:
        x_n = x.get("param_name").upper()
        x_fmt = x.get("format")
        x_dtype = x.get("dtype")
        options.append("-DDTYPE_{n}={t}".format(n=x_n, t=DTYPE_MAP.get(x_dtype)[1]))
        options.append("-DORIG_DTYPE_{n}={ot}".format(n=x_n, ot=DTYPE_MAP.get(x_dtype)[0]))
        options.append("-DFORMAT_{n}=FORMAT_{f}".format(n=x_n, f=x_fmt))
    return options

def load_dso(so_path):
    try:
        ctypes.CDLL(so_path)
    except OSError as error :
        print(error)
        raise RuntimeError("cannot open %s" %(so_path))
    else:
        print("load so succ ", so_path)

def get_shortsoc_compile_option(compile_option_list: list, shortsoc:str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options = compile_option_list[shortsoc]
    elif '__ALLSOC__' in compile_option_list:
        compile_options = compile_option_list['__ALLSOC__']
    return compile_options

def _build_args(x, z):
    __inputs__ = []
    for arg in [x]:
        if arg != None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __inputs__.append(arg[0])
            else:
                __inputs__.append(arg)
    __outputs__ = []
    for arg in [z]:
        if arg != None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __outputs__.append(arg[0])
            else:
                __outputs__.append(arg)
    __attrs__ = []
    return __inputs__, __outputs__, __attrs__

@tbe_register.register_operator("SinhCustom")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sinh_custom(x, z, kernel_name="sinh_custom", impl_mode=""):
    if get_current_build_config("enable_op_prebuild"):
        return
    __inputs__, __outputs__, __attrs__ = _build_args(x, z)
    options = get_dtype_fmt_options(__inputs__, __outputs__)
    options += ["-x", "cce"]
    ccec = os.environ.get('CCEC_REAL_PATH')
    if ccec is None:
        ccec = shutil.which("ccec")
    if ccec != None:
        ccec_path = os.path.dirname(ccec)
        tikcpp_path = os.path.realpath(os.path.join(ccec_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath("/usr/local/Ascend/latest/compiler/tikcpp")
    options.append("-I" + tikcpp_path)
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common"))
    if impl_mode == "high_performance":
        options.append("-DHIGH_PERFORMANCE=1")
    elif impl_mode == "high_precision":
        options.append("-DHIGH_PRECISION=1")
    if get_default_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETEMINISTIC_MODE=1")

    custom_compile_options = {},
    custom_all_compile_options = {},
    soc_version = get_soc_spec("SOC_VERSION")
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    custom_compile_options_soc = get_shortsoc_compile_option(custom_compile_options[0], soc_short)
    custom_all_compile_options_soc = get_shortsoc_compile_option(custom_all_compile_options[0], soc_short)
    options += custom_all_compile_options_soc
    options += custom_compile_options_soc

    origin_func_name = "sinh_custom"
    ascendc_src_dir = "sinh_custom"
    ascendc_src_file = "sinh_custom.cpp"
    src = os.path.join(PYF_PATH, "..", "ascendc", ascendc_src_dir, ascendc_src_file)
    if not os.path.exists(src):
        src = os.path.join(PYF_PATH, ascendc_src_file)

    print("start compile Ascend C operator SinhCustom. kernel name is sinh_custom")
    op_type = "SinhCustom"
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(kernel_name = kernel_name, op_type = op_type, inputs = __inputs__, outputs = __outputs__,\
        attrs = __attrs__, impl_mode = impl_mode, origin_inputs=[x], origin_outputs = [z])
    compile_op(src, origin_func_name, op_info, options, code_channel, '{}')

def op_select_format(x, z, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args(x, z)
    result = check_op_cap("op_select_format", "SinhCustom", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")

def get_op_specific_info(x, z, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args(x, z)
    result = check_op_cap("get_op_specific_info", "SinhCustom", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")
