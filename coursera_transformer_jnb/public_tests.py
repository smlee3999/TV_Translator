import numpy as np
import tensorflow as tf

# Google Colab (TensorFlow 2.16+ / Keras 3) compatible version.
#
# Keras 3 no longer honors tf.random.set_seed() for weight initialization and
# uses a different initialization/RNG scheme, so the hard-coded expected values
# of the original tests cannot be reproduced. Layer tests therefore compare the
# student's output (training=False) against a reference forward pass built from
# the very same sub-layers (and therefore the same weights), and check
# shapes, masking, LayerNormalization statistics and dropout behavior.
#
# Keras 3 also rejects non-tensor positional arguments (e.g. `training`), so
# layers are called with keyword arguments: layer(x, training=..., mask=...).


def _set_seed(seed=10):
    tf.keras.utils.set_random_seed(seed)


def _np(x):
    return x.numpy() if tf.is_tensor(x) else np.asarray(x)


def _check_layernorm(out, msg):
    # Freshly created LayerNormalization layers have gamma=1, beta=0
    out = _np(out)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-4), msg
    assert np.allclose(out.std(axis=-1), 1, atol=1e-2), msg


def _ref_encoder_layer(layer, x, mask):
    attn = layer.mha(x, x, x, attention_mask=mask, training=False)
    skip = layer.layernorm1(x + attn)
    return layer.layernorm2(skip + layer.ffn(skip))


def _ref_encoder(enc, x, mask):
    seq_len = x.shape[1]
    out = enc.embedding(x) * np.sqrt(enc.embedding_dim)
    out = out + enc.pos_encoding[:, :seq_len, :]
    for layer in enc.enc_layers:
        out = _ref_encoder_layer(layer, out, mask)
    return out


def _ref_decoder_layer(layer, x, enc_output, look_ahead_mask, padding_mask):
    attn1, w1 = layer.mha1(x, x, x, attention_mask=look_ahead_mask,
                           return_attention_scores=True, training=False)
    q1 = layer.layernorm1(x + attn1)
    attn2, w2 = layer.mha2(q1, enc_output, enc_output, attention_mask=padding_mask,
                           return_attention_scores=True, training=False)
    out2 = layer.layernorm2(q1 + attn2)
    out3 = layer.layernorm3(out2 + layer.ffn(out2))
    return out3, w1, w2


def _ref_decoder(dec, x, enc_output, look_ahead_mask, padding_mask):
    seq_len = x.shape[1]
    out = dec.embedding(x) * np.sqrt(dec.embedding_dim)
    out = out + dec.pos_encoding[:, :seq_len, :]
    weights = []
    for layer in dec.dec_layers:
        out, w1, w2 = _ref_decoder_layer(layer, out, enc_output, look_ahead_mask, padding_mask)
        weights += [w1, w2]
    return out, weights


def get_angles_test(target):
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = target(pos_m, dims, d_model)

    assert type(result) == np.ndarray, "You must return a numpy ndarray"
    assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols =  result[:, 0::2]
    odd_cols = result[:,  1::2]
    assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
    limit = (position - 1) / np.power(10000,14.0/16.0)
    assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"

    print("\033[92mAll tests passed")

def positional_encoding_test(target, get_angles):
    position = 8
    d_model = 16

    pos_encoding = target(position, d_model)

    assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

    sin_part = pos_encoding.numpy()[:, :, 0::2]
    cos_part = pos_encoding.numpy()[:, :, 1::2]

    ones = sin_part ** 2  +  cos_part ** 2
    assert np.allclose(ones, np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"

    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part < 0] += np.pi
    angs = angs % (2 * np.pi)

    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)

    assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"

    print("\033[92mAll tests passed")

def scaled_dot_product_attention_test(target):
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    attention, weights = target(q, k, v, None)
    assert tf.is_tensor(weights), "Weights must be a tensor"
    assert tuple(tf.shape(weights).numpy()) == (q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(weights, [[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                   [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                   [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]]), "Wrong unmasked weights"

    assert tf.is_tensor(attention), "Output must be a tensor"
    assert tuple(tf.shape(attention).numpy()) == (q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(attention, [[0.74105227, 0.15705977],
                                   [0.7227253,  0.16817567],
                                   [0.6637989,  0.2039163 ]]), "Wrong unmasked attention"

    mask = np.array([[[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]])
    attention, weights = target(q, k, v, mask)

    assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]), "Wrong masked weights"
    assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                   [0.6163482666015625, 0.2326965481042862],
                                   [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"

    print("\033[92mAll tests passed")

def EncoderLayer_test(target):
    _set_seed(10)
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = target(4, 2, 8)
    mask = np.array([[1, 0, 1]])

    encoded = encoder_layer1(q, training=True, mask=mask)

    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"
    _check_layernorm(encoded, "Output must be normalized. Did you apply layernorm2 at the end?")

    expected = _ref_encoder_layer(encoder_layer1, q, mask)
    assert not np.allclose(encoded.numpy(), expected.numpy()), "Output with training=True must use dropout. Are you passing training to dropout_ffn?"

    encoded = encoder_layer1(q, training=False, mask=mask)
    assert np.allclose(encoded.numpy(), expected.numpy(), atol=1e-5), "Wrong values when training=False"

    mask = np.array([[1, 1, 0]])
    encoded = encoder_layer1(q, training=False, mask=mask)
    expected = _ref_encoder_layer(encoder_layer1, q, mask)
    assert np.allclose(encoded.numpy(), expected.numpy(), atol=1e-5), "Wrong values when training=False. Are you passing the mask to self.mha?"
    print("\033[92mAll tests passed")

def Encoder_test(target):
    _set_seed(10)

    embedding_dim=4

    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)

    x = np.array([[2, 1, 3], [1, 2, 0]])

    encoderq_output = encoderq(x, training=True, mask=None)

    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (x.shape[0], x.shape[1], embedding_dim), f"Wrong shape. We expected ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
    _check_layernorm(encoderq_output, "Wrong values case 1")

    encoderq_output = encoderq(x, training=False, mask=None)
    expected = _ref_encoder(encoderq, x, None)
    assert np.allclose(encoderq_output.numpy(), expected.numpy(), atol=1e-5), "Wrong values case 2 (training=False, no mask)"

    mask = np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]])
    encoderq_output = encoderq(x, training=True, mask=mask)
    expected = _ref_encoder(encoderq, x, mask)
    assert not np.allclose(encoderq_output.numpy(), expected.numpy()), "Output with training=True must use dropout. Are you passing training to the dropout and encoder layers?"

    encoderq_output = encoderq(x, training=False, mask=mask)
    assert np.allclose(encoderq_output.numpy(), expected.numpy(), atol=1e-5), "Wrong values case 3 (training=False, with mask)"

    print("\033[92mAll tests passed")

def DecoderLayer_test(target, create_look_ahead_mask):

    num_heads=8
    _set_seed(10)

    decoderLayerq = target(
        embedding_dim=4,
        num_heads=num_heads,
        fully_connected_dim=32,
        dropout_rate=0.1,
        layernorm_eps=1e-6)

    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                                   [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                                   [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]]])

    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)

    look_ahead_mask = create_look_ahead_mask(q.shape[1])

    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"

    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(out).numpy()) == q.shape, f"Wrong shape. We expected {q.shape}"
    _check_layernorm(out, "Wrong values in out. Did you apply layernorm3 at the end?")

    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    exp_out, exp_w1, exp_w2 = _ref_decoder_layer(decoderLayerq, q, encoderq_output, look_ahead_mask, padding_mask)

    assert np.allclose(np.triu(attn_w_b1.numpy()[0, 0], k=1), 0), "Wrong values in attn_w_b1. Check the call to self.mha1 (look_ahead_mask)"
    assert np.allclose(attn_w_b1, exp_w1, atol=1e-5), "Wrong values in attn_w_b1. Check the call to self.mha1"
    assert np.allclose(attn_w_b2, exp_w2, atol=1e-5),  "Wrong values in attn_w_b2. Check the call to self.mha2"
    assert np.allclose(out, exp_out, atol=1e-5), "Wrong values in out"


    # Now let's try a example with padding mask
    padding_mask = np.array([[[1, 1, 0]]])
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    exp_out, exp_w1, exp_w2 = _ref_decoder_layer(decoderLayerq, q, encoderq_output, look_ahead_mask, padding_mask)
    assert np.allclose(attn_w_b2.numpy()[..., 2], 0), "Wrong values in attn_w_b2 when we mask the last word. Are you passing the padding_mask to self.mha2?"
    assert np.allclose(out, exp_out, atol=1e-5), "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"

    print("\033[92mAll tests passed")

def Decoder_test(target, create_look_ahead_mask, create_padding_mask):
    _set_seed(10)

    num_layers=7
    embedding_dim=4
    num_heads=3
    fully_connected_dim=8
    target_vocab_size=33
    maximum_position_encoding=6

    x_array = np.array([[3, 2, 1], [2, 1, 0]])


    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]])

    look_ahead_mask = create_look_ahead_mask(x_array.shape[1])

    decoderk = target(num_layers,
                    embedding_dim,
                    num_heads,
                    fully_connected_dim,
                    target_vocab_size,
                    maximum_position_encoding)
    x, attention_weights = decoderk(x_array, encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)
    assert tf.is_tensor(x), "Wrong type for x. It must be a tensor"
    assert np.allclose(tf.shape(x), tf.shape(encoderq_output)), f"Wrong shape. We expected { tf.shape(encoderq_output)}"

    exp_x, exp_weights = _ref_decoder(decoderk, x_array, encoderq_output, look_ahead_mask, None)
    assert np.allclose(x, exp_x, atol=1e-5), "Wrong values in x"

    assert type(attention_weights) == dict, "Wrong type for attention_weights. It must be a dict"
    keys = list(attention_weights.keys())
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(attention_weights[keys[0]]), f"Wrong type for attention_weights[{keys[0]}]. Output must be a tensor"
    shape1 = (x_array.shape[0], num_heads, x_array.shape[1], x_array.shape[1])
    assert tuple(tf.shape(attention_weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    for key, exp_w in zip(keys, exp_weights):
        assert np.allclose(attention_weights[key], exp_w, atol=1e-5), f"Wrong values in attention_weights[{key}]"

    x, attention_weights = decoderk(x_array, encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=None)
    assert not np.allclose(x, exp_x), "Wrong values in x when training=True. Are you passing training to the dropout and decoder layers?"

    padding_mask = create_padding_mask(x_array)
    x, attention_weights = decoderk(x_array, encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    exp_x, exp_weights = _ref_decoder(decoderk, x_array, encoderq_output, look_ahead_mask, padding_mask)
    assert np.allclose(x, exp_x, atol=1e-5), "Wrong values in x when use padding mask"

    print("\033[92mAll tests passed")

def Transformer_test(target, create_look_ahead_mask, create_padding_mask):

    _set_seed(10)


    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = target(num_layers,
                        embedding_dim,
                        num_heads,
                        fully_connected_dim,
                        input_vocab_size,
                        target_vocab_size,
                        max_positional_encoding_input,
                        max_positional_encoding_target)
    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = create_padding_mask(sentence_lang_a)
    dec_padding_mask = create_padding_mask(sentence_lang_b)

    look_ahead_mask = create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        training=True,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )


    assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert tuple(tf.shape(translation).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert np.allclose(tf.reduce_sum(translation, axis=-1), 1), "Wrong values in translation. Did you apply the final softmax layer?"

    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"

    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert np.allclose(weights[keys[0]].numpy()[0, 0, 1, 2:], 0), f"Wrong values in weights[{keys[0]}]. Are you passing look_ahead_mask to the decoder?"

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        training=False,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )

    enc_output = _ref_encoder(trans.encoder, sentence_lang_a, enc_padding_mask)
    dec_output, exp_weights = _ref_decoder(trans.decoder, sentence_lang_b, enc_output, look_ahead_mask, dec_padding_mask)
    expected = trans.final_layer(dec_output)
    assert np.allclose(translation, expected, atol=1e-5), "Wrong values in outd"
    for key, exp_w in zip(weights.keys(), exp_weights):
        assert np.allclose(weights[key], exp_w, atol=1e-5), f"Wrong values in weights[{key}]"

    print(translation)

    print("\033[92mAll tests passed")
