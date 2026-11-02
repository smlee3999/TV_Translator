"""
학습 시 GPU 메모리 폭증(OOM)을 막기 위한 BatchMatMulV2 gradient 교체.

Keras 3 의 TensorFlow 백엔드는 Dense / EinsumDense 를 tf.matmul(x[B,T,K], w[K,N]) 처럼
3차원 입력과 2차원 가중치의 broadcast matmul 로 계산한다. TensorFlow 의 기본 gradient 는
가중치 gradient 를 배치마다 [B,K,N] 으로 만든 뒤 합산하므로, 출력층(256x16000)에서만
배치 682 기준 약 11GB 가 필요하다 (OOM when allocating tensor with shape[682,256,16000]).

여기서는 그 경우에만 x 를 [B*T,K] 로 펼쳐 [K,N] gradient 를 바로 계산한다.
결과 값은 기본 gradient 와 같고, 다른 경우는 원래 gradient 함수를 그대로 쓴다.
"""
import tensorflow as tf
from tensorflow.python.framework import ops

_OP = "BatchMatMulV2"
_installed = False


def install():
    global _installed
    if _installed:
        return
    entry = ops._gradient_registry._registry[_OP]  # pylint: disable=protected-access
    original = entry["type"]

    def grad(op, dz):
        x, y = op.inputs
        if (x.shape.rank is not None and x.shape.rank >= 3 and y.shape.rank == 2
                and not op.get_attr("adj_x") and not op.get_attr("adj_y")):
            k, n = tf.shape(y)[0], tf.shape(y)[1]
            dx = tf.matmul(dz, y, transpose_b=True)
            dy = tf.matmul(tf.reshape(x, [-1, k]), tf.reshape(dz, [-1, n]), transpose_a=True)
            return dx, dy
        return original(op, dz)

    entry["type"] = grad
    _installed = True
