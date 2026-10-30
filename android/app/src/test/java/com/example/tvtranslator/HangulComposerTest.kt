package com.example.tvtranslator

import org.junit.Assert.assertEquals
import org.junit.Test

class HangulComposerTest {

    /** 영문 키 시퀀스를 두벌식으로 입력한 결과 (backspace 는 '<') */
    private fun type(keys: String): String {
        val composer = HangulComposer()
        val sb = StringBuilder()
        for (k in keys) {
            when {
                k == '<' -> if (!composer.backspace() && sb.isNotEmpty()) sb.setLength(sb.length - 1)
                k == ' ' -> sb.append(composer.commit()).append(' ')
                else -> sb.append(composer.input(HangulComposer.jamoForKey(k, k.isUpperCase())!!))
            }
        }
        return sb.append(composer.commit()).toString()
    }

    @Test fun basicSyllables() {
        assertEquals("안녕하세요", type("dkssudgktpdy"))
        assertEquals("한글 입력", type("gksrmf dlqfur"))
    }

    @Test fun compoundVowelsAndFinals() {
        assertEquals("와", type("dhk"))
        assertEquals("의사", type("dmltk"))
        assertEquals("닭", type("ekfr"))
        assertEquals("값", type("rkqt"))
        assertEquals("읽어", type("dlfrdj"))
    }

    @Test fun finalConsonantMovesToNextSyllable() {
        assertEquals("달가", type("ekfrk"))
        assertEquals("가사", type("rktk"))
        assertEquals("갑시", type("rkqtl"))
    }

    @Test fun shiftAndDoubleConsonants() {
        assertEquals("빠른", type("Qkfms"))
        assertEquals("있어", type("dlTdj"))
        assertEquals("얘기", type("dOrl"))
    }

    @Test fun standaloneJamo() {
        assertEquals("ㅋㅋㅋ", type("zzz"))
        assertEquals("ㅏㅏ", type("kk"))
        assertEquals("ㄸ", type("E"))
    }

    @Test fun backspaceRemovesLastJamo() {
        assertEquals("달", type("ekfr<"))
        assertEquals("다", type("ekfr<<"))
        assertEquals("ㄷ", type("ekfr<<<"))
        assertEquals("가ㅅ", type("rktk<"))
        assertEquals("ㅗ", type("hk<"))
    }
}
