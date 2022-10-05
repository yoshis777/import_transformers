# https://self-development.info/pycuda%e3%81%ae%e3%82%a4%e3%83%b3%e3%82%b9%e3%83%88%e3%83%bc%e3%83%ab%e3%80%90python-on-windows%e3%80%91/
# GPUにデータを転送するプログラム
 
# PyCudaのインポート
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
# numpy配列用
import numpy
 
# 4x4の乱数のnumpy配列を作成
a = numpy.random.randn(4,4)
 
# nVidiaデバイス（GPU）は単精度のみサポート→変換
a = a.astype(numpy.float32)
 
# GPU上に送信データ用のメモリを割り当てる
a_gpu = cuda.mem_alloc(a.nbytes)
 
# GPUにデータを転送
cuda.memcpy_htod(a_gpu, a)
