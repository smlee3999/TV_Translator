package com.example.tvtranslator

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 한↔영 양방향 Transformer(TFLite) 번역기.
 *
 * training/export_tflite.py 의 TFLiteTranslator 와 같은 로직이다.
 *   문장 분리 → SentencePiece 토큰화 → [목표언어 태그] + ids + [EOS]
 *   → encode 1회 → decode 반복(greedy) → detokenize
 *
 * Interpreter 는 thread-safe 하지 않으므로 한 스레드에서만 호출해야 한다.
 */
class Translator private constructor(
    private val interpreter: Interpreter,
    private val tokenizer: SpmTokenizer,
    private val config: ModelConfig,
) : Closeable {

    enum class Target { KO, EN }

    fun translate(text: String, target: Target): String {
        val tag = if (target == Target.KO) config.toKoId else config.toEnId
        val chunk = config.maxTokens - 2  // 태그와 EOS 자리
        val results = ArrayList<String>()
        for (sentence in SENTENCE_SPLIT.split(text)) {
            val ids = tokenizer.encode(sentence)
            var i = 0
            while (i < ids.size) {
                val end = minOf(i + chunk, ids.size)
                val src = IntArray(end - i + 2)
                src[0] = tag
                ids.copyInto(src, 1, i, end)
                src[src.size - 1] = config.eosId
                results += tokenizer.decode(translateIds(src))
                i = end
            }
        }
        return results.filter { it.isNotEmpty() }.joinToString(" ")
    }

    private fun translateIds(src: IntArray): IntArray {
        val encOutput = Array(1) { Array(src.size) { FloatArray(config.embeddingDim) } }
        interpreter.runSignature(mapOf("src_ids" to arrayOf(src)), mapOf("enc_output" to encOutput), "encode")

        val out = ArrayList<Int>().apply { add(config.bosId) }
        val maxOut = minOf(config.maxTokens, 2 * src.size + 10)  // train.max_output_len 과 동일
        val nextId = IntArray(1)
        while (out.size < maxOut) {
            interpreter.runSignature(
                mapOf("enc_output" to encOutput, "tgt_ids" to arrayOf(out.toIntArray())),
                mapOf("next_id" to nextId),
                "decode",
            )
            if (nextId[0] == config.eosId) break
            out += nextId[0]
        }
        return out.subList(1, out.size).toIntArray()
    }

    override fun close() = interpreter.close()

    data class ModelConfig(
        val bosId: Int,
        val eosId: Int,
        val toEnId: Int,
        val toKoId: Int,
        val maxTokens: Int,
        val embeddingDim: Int,
    )

    companion object {
        const val MODEL_ASSET = "translator.tflite"
        const val VOCAB_ASSET = "tokenizer.vocab"
        const val CONFIG_ASSET = "model_config.json"

        private val SENTENCE_SPLIT = Regex("(?<=[.!?。？！])\\s+|\\n+")

        /** assets 의 모델/vocab/config 를 읽어 번역기를 만든다. 파일이 없으면 IOException. */
        fun create(context: Context, numThreads: Int = 4): Translator {
            val assets = context.assets
            val json = JSONObject(assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() })
            val config = ModelConfig(
                bosId = json.getInt("bos_id"),
                eosId = json.getInt("eos_id"),
                toEnId = json.getInt("to_en_id"),
                toKoId = json.getInt("to_ko_id"),
                maxTokens = json.getInt("max_tokens"),
                embeddingDim = json.getInt("embedding_dim"),
            )
            val tokenizer = assets.open(VOCAB_ASSET).use { SpmTokenizer.load(it) }
            val interpreter = Interpreter(loadModel(context), Interpreter.Options().setNumThreads(numThreads))
            return Translator(interpreter, tokenizer, config)
        }

        private fun loadModel(context: Context): MappedByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }

        /** 한글이 들어 있으면 한→영, 아니면 영→한 */
        fun detectTarget(text: String): Target =
            if (text.any { it in '가'..'힣' || it in 'ㄱ'..'ㆎ' }) Target.EN else Target.KO
    }
}
