package com.example.tvtranslator

import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.text.Normalizer

/**
 * SentencePiece unigram 토크나이저 (byte_fallback 지원).
 *
 * training/spm_tokenizer.py 의 UnigramTokenizer 와 같은 알고리즘이며,
 * export_tflite.py 가 만든 tokenizer.vocab (piece<TAB>score<TAB>type) 만으로 동작한다.
 * type: n=일반, u=unknown, c=control, d=user defined(<2en>/<2ko>), b=byte, x=unused
 */
class SpmTokenizer(
    private val pieces: List<String>,
    private val scores: FloatArray,
    private val types: CharArray,
) {
    private class TrieNode {
        val children = HashMap<Char, TrieNode>()
        var id = -1
    }

    private val root = TrieNode()
    private val unkId: Int
    private val unkScore: Float
    private val byteToId = IntArray(256) { -1 }
    private val hasByteFallback: Boolean

    init {
        var minScore = Float.MAX_VALUE
        var unk = -1
        for (i in pieces.indices) {
            when (types[i]) {
                'n' -> {
                    var node = root
                    for (c in pieces[i]) node = node.children.getOrPut(c) { TrieNode() }
                    node.id = i
                    if (scores[i] < minScore) minScore = scores[i]
                }
                'u' -> unk = i
                'b' -> byteToId[pieces[i].substring(3, 5).toInt(16)] = i
            }
        }
        unkId = unk
        unkScore = minScore - UNK_PENALTY
        hasByteFallback = byteToId.all { it >= 0 }
    }

    fun encode(text: String): IntArray {
        val normalized = TextNormalizer.normalize(text)
        if (normalized.isEmpty()) return IntArray(0)
        val s = WS + normalized.replace(' ', WS)
        val n = s.length
        val best = FloatArray(n + 1)
        val reached = BooleanArray(n + 1).also { it[0] = true }
        val backStart = IntArray(n + 1)
        val backId = IntArray(n + 1)

        var start = 0
        while (start < n) {
            if (!reached[start]) {
                start++
                continue
            }
            val oneChar = Character.charCount(s.codePointAt(start))
            var hasSingle = false
            var node: TrieNode? = root
            var end = start
            // 짧은 piece 부터 검사하고, 점수가 '더 클 때만' 갱신한다 (sentencepiece 와 같은 동점 처리)
            while (end < n) {
                node = node!!.children[s[end]] ?: break
                end++
                val id = node.id
                if (id < 0) continue
                if (end - start == oneChar) hasSingle = true
                val cand = best[start] + scores[id]
                if (!reached[end] || cand > best[end]) {
                    best[end] = cand
                    reached[end] = true
                    backStart[end] = start
                    backId[end] = id
                }
            }
            if (!hasSingle) {
                val e = start + oneChar
                val cand = best[start] + unkScore
                if (!reached[e] || cand > best[e]) {
                    best[e] = cand
                    reached[e] = true
                    backStart[e] = start
                    backId[e] = unkId
                }
            }
            start++
        }

        val segments = ArrayList<IntArray>()  // (start, end, id)
        var pos = n
        while (pos > 0) {
            segments += intArrayOf(backStart[pos], pos, backId[pos])
            pos = backStart[pos]
        }
        val ids = ArrayList<Int>(segments.size)
        for (k in segments.indices.reversed()) {
            val (st, en, id) = segments[k]
            if (id == unkId && hasByteFallback) {
                for (b in s.substring(st, en).toByteArray(Charsets.UTF_8)) ids += byteToId[b.toInt() and 0xFF]
            } else {
                ids += id
            }
        }
        return ids.toIntArray()
    }

    fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        val bytes = ByteArrayOutputStream()
        fun flushBytes() {
            if (bytes.size() > 0) {
                sb.append(String(bytes.toByteArray(), Charsets.UTF_8))
                bytes.reset()
            }
        }
        for (id in ids) {
            if (id < 0 || id >= pieces.size) continue
            when (types[id]) {
                'b' -> {
                    bytes.write(pieces[id].substring(3, 5).toInt(16))
                    continue
                }
                'n' -> {
                    flushBytes()
                    sb.append(pieces[id])
                }
                'u' -> {
                    flushBytes()
                    sb.append(" ⁇ ")
                }
                else -> flushBytes()
            }
        }
        flushBytes()
        return TextNormalizer.strip(sb.toString().replace(WS, ' '))
    }

    companion object {
        private const val WS = '▁'
        private const val UNK_PENALTY = 10.0f

        fun load(input: InputStream): SpmTokenizer {
            val pieces = ArrayList<String>()
            val scores = ArrayList<Float>()
            val types = ArrayList<Char>()
            input.bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (line in lines) {
                    if (line.isEmpty()) continue
                    val cols = line.split('\t')
                    pieces += cols[0]
                    scores += cols[1].toFloat()
                    types += cols[2][0]
                }
            }
            return SpmTokenizer(pieces, scores.toFloatArray(), types.toCharArray())
        }
    }
}

/** training/spm_tokenizer.py 의 normalize_text 와 동일: NFKC + 공백 정리. */
object TextNormalizer {
    /** Python str.isspace() 와 같은 기준 */
    fun isSpace(c: Char): Boolean = Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'

    fun normalize(text: String): String {
        val nfkc = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val sb = StringBuilder(nfkc.length)
        var pendingSpace = false
        for (c in nfkc) {
            if (isSpace(c)) {
                pendingSpace = sb.isNotEmpty()
            } else {
                if (pendingSpace) sb.append(' ')
                pendingSpace = false
                sb.append(c)
            }
        }
        return sb.toString()
    }

    fun strip(text: String): String = text.trim { isSpace(it) }
}
