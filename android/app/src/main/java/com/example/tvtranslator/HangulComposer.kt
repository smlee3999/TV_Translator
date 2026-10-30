package com.example.tvtranslator

/**
 * 두벌식 한글 오토마타.
 *
 * Android TV 는 블루투스/USB 물리 키보드로 한글을 조합해 주는 IME 가 없는 기기가 많아서
 * 앱이 직접 자모를 조합한다. input() 은 이번 입력으로 '확정된' 문자열을 돌려주고,
 * 아직 조합 중인 글자는 [composing] 으로 얻는다.
 */
class HangulComposer {

    private data class State(val cho: Int = -1, val jung: Int = -1, val jong: Int = 0)

    // 현재 조합 중인 글자의 입력 이력. backspace 는 마지막 자모 입력을 취소한다.
    private val history = ArrayList<State>()
    private val state: State get() = history.lastOrNull() ?: State()

    val isComposing: Boolean get() = history.isNotEmpty()

    val composing: String get() = if (history.isEmpty()) "" else render(state)

    /** 자모 하나를 입력한다. 반환값은 이번 입력으로 확정된 문자열. */
    fun input(jamo: Char): String {
        val s = state
        val choIdx = CHO.indexOf(jamo)
        val jungIdx = JUNG.indexOf(jamo)
        return when {
            jungIdx >= 0 -> inputVowel(s, jungIdx)
            choIdx >= 0 -> inputConsonant(s, jamo, choIdx)
            else -> commit() + jamo
        }
    }

    private fun inputConsonant(s: State, jamo: Char, choIdx: Int): String {
        val jongIdx = JONG.indexOf(jamo)
        when {
            !isComposing -> {}
            s.cho >= 0 && s.jung >= 0 && s.jong == 0 && jongIdx > 0 -> {
                history += s.copy(jong = jongIdx)
                return ""
            }
            s.jong > 0 -> {
                val combined = JONG_COMBINE[JONG[s.jong] to jamo]
                if (combined != null) {
                    history += s.copy(jong = JONG.indexOf(combined))
                    return ""
                }
            }
        }
        val committed = commit()
        history += State(cho = choIdx)
        return committed
    }

    private fun inputVowel(s: State, jungIdx: Int): String {
        if (!isComposing) {
            history += State(jung = jungIdx)
            return ""
        }
        if (s.jong > 0) {
            // 받침을 다음 글자의 초성으로 옮긴다 (닭+ㅏ → 달가, 각+ㅏ → 가가)
            val split = JONG_SPLIT[JONG[s.jong]]
            val (remain, moved) = if (split != null) JONG.indexOf(split.first) to split.second
            else 0 to JONG[s.jong]
            val committed = render(s.copy(jong = remain))
            history.clear()
            history += State(cho = CHO.indexOf(moved))
            history += State(cho = CHO.indexOf(moved), jung = jungIdx)
            return committed
        }
        if (s.jung < 0) {
            history += s.copy(jung = jungIdx)
            return ""
        }
        val combined = JUNG_COMBINE[JUNG[s.jung] to JUNG[jungIdx]]
        if (combined != null) {
            history += s.copy(jung = JUNG.indexOf(combined))
            return ""
        }
        val committed = commit()
        history += State(jung = jungIdx)
        return committed
    }

    /** 마지막 자모 입력을 취소한다. 조합 중이 아니면 false. */
    fun backspace(): Boolean {
        if (history.isEmpty()) return false
        history.removeAt(history.size - 1)
        return true
    }

    /** 조합 중인 글자를 확정하고 돌려준다. */
    fun commit(): String {
        val text = composing
        history.clear()
        return text
    }

    private fun render(s: State): String = when {
        s.cho >= 0 && s.jung >= 0 -> (0xAC00 + (s.cho * 21 + s.jung) * 28 + s.jong).toChar().toString()
        s.cho >= 0 -> CHO[s.cho].toString()
        s.jung >= 0 -> JUNG[s.jung].toString()
        else -> ""
    }

    companion object {
        private const val CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
        private const val JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
        private const val JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

        private val JUNG_COMBINE = mapOf(
            ('ㅗ' to 'ㅏ') to 'ㅘ', ('ㅗ' to 'ㅐ') to 'ㅙ', ('ㅗ' to 'ㅣ') to 'ㅚ',
            ('ㅜ' to 'ㅓ') to 'ㅝ', ('ㅜ' to 'ㅔ') to 'ㅞ', ('ㅜ' to 'ㅣ') to 'ㅟ',
            ('ㅡ' to 'ㅣ') to 'ㅢ',
        )
        private val JONG_COMBINE = mapOf(
            ('ㄱ' to 'ㅅ') to 'ㄳ', ('ㄴ' to 'ㅈ') to 'ㄵ', ('ㄴ' to 'ㅎ') to 'ㄶ',
            ('ㄹ' to 'ㄱ') to 'ㄺ', ('ㄹ' to 'ㅁ') to 'ㄻ', ('ㄹ' to 'ㅂ') to 'ㄼ',
            ('ㄹ' to 'ㅅ') to 'ㄽ', ('ㄹ' to 'ㅌ') to 'ㄾ', ('ㄹ' to 'ㅍ') to 'ㄿ',
            ('ㄹ' to 'ㅎ') to 'ㅀ', ('ㅂ' to 'ㅅ') to 'ㅄ',
        )
        private val JONG_SPLIT = JONG_COMBINE.entries.associate { (pair, combined) -> combined to pair }

        // 두벌식 자판: QWERTY 키 → 자모 (대문자는 Shift)
        private const val KEYS = "qwertyuiopasdfghjklzxcvbnm"
        private const val JAMO = "ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ"
        private const val SHIFT_JAMO = "ㅃㅉㄸㄲㅆㅛㅕㅑㅒㅖㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ"

        /** 영문 키(a~z)와 Shift 상태를 두벌식 자모로 바꾼다. 해당 없으면 null. */
        fun jamoForKey(letter: Char, shift: Boolean): Char? {
            val i = KEYS.indexOf(letter.lowercaseChar())
            if (i < 0) return null
            return if (shift) SHIFT_JAMO[i] else JAMO[i]
        }
    }
}
