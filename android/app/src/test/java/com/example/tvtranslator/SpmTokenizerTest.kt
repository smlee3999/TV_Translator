package com.example.tvtranslator

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeNotNull
import org.junit.Test

/**
 * training/export_tflite.py 가 sentencepiece 결과로 만든 tokenizer_testcases.tsv 와
 * Kotlin 토크나이저 결과가 같은지 확인한다. (export 시 src/test/resources 로 복사됨)
 */
class SpmTokenizerTest {

    private fun resource(name: String) = javaClass.classLoader?.getResourceAsStream(name)

    @Test fun matchesSentencePiece() {
        val vocab = resource("tokenizer.vocab")
        val cases = resource("tokenizer_testcases.tsv")
        assumeNotNull(vocab, cases)
        val tokenizer = vocab!!.use { SpmTokenizer.load(it) }
        var count = 0
        cases!!.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (line.isEmpty()) continue
                val (text, ids) = line.split('\t')
                val expected = if (ids.isEmpty()) IntArray(0) else ids.split(' ').map { it.toInt() }.toIntArray()
                assertArrayEquals("encode: $text", expected, tokenizer.encode(text))
                assertEquals("decode: $text", TextNormalizer.normalize(text), tokenizer.decode(expected))
                count++
            }
        }
        println("tokenizer test cases: $count")
    }

    @Test fun normalizer() {
        assertEquals("hello world", TextNormalizer.normalize("  hello \t\n world  "))
        assertEquals("Hello world", TextNormalizer.normalize("Ｈｅｌｌｏ　ｗｏｒｌｄ"))
        assertEquals("", TextNormalizer.normalize("   "))
    }
}
