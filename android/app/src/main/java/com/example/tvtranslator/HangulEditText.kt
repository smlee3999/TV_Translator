package com.example.tvtranslator

import android.content.Context
import android.text.Spanned
import android.text.style.UnderlineSpan
import android.util.AttributeSet
import android.view.InputDevice
import android.view.KeyEvent
import androidx.appcompat.widget.AppCompatEditText

/**
 * 물리 키보드(블루투스/USB)에서 두벌식 한글 입력을 지원하는 EditText.
 *
 * 한글 모드에서는 영문 키 입력을 IME 보다 먼저(onKeyPreIme) 가로채 [HangulComposer] 로 조합한다.
 * 리모컨 D-pad 와 화상 키보드(IME) 입력은 그대로 통과시킨다.
 * 한/영 전환: 한/영 키, 오른쪽 Alt, Shift+Space
 */
class HangulEditText @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = androidx.appcompat.R.attr.editTextStyle,
) : AppCompatEditText(context, attrs, defStyleAttr) {

    private val composer = HangulComposer()
    private val composingSpan = UnderlineSpan()
    private var composeStart = -1
    private var updating = false

    // TextView 생성자 안에서 onSelectionChanged 등이 호출될 때는 아직 false (필드 초기화 전)
    private var initialized = true

    var onHangulModeChanged: ((Boolean) -> Unit)? = null

    var hangulMode = false
        set(value) {
            if (field == value) return
            finishComposing()
            field = value
            onHangulModeChanged?.invoke(value)
        }

    override fun onKeyPreIme(keyCode: Int, event: KeyEvent): Boolean {
        if (!isFromPhysicalKeyboard(event)) return super.onKeyPreIme(keyCode, event)

        if (isToggleKey(event)) {
            if (event.action == KeyEvent.ACTION_DOWN && event.repeatCount == 0) hangulMode = !hangulMode
            return true
        }
        if (!hangulMode) return super.onKeyPreIme(keyCode, event)

        val letter = if (keyCode in KeyEvent.KEYCODE_A..KeyEvent.KEYCODE_Z && (event.metaState and CTRL_ALT_META) == 0)
            'a' + (keyCode - KeyEvent.KEYCODE_A) else null
        val jamo = letter?.let { HangulComposer.jamoForKey(it, event.isShiftPressed) }
        when {
            jamo != null -> {
                if (event.action == KeyEvent.ACTION_DOWN) applyComposition(composer.input(jamo))
                return true
            }
            keyCode == KeyEvent.KEYCODE_DEL && composer.isComposing -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    composer.backspace()
                    applyComposition("")
                }
                return true
            }
            event.action == KeyEvent.ACTION_DOWN && !KeyEvent.isModifierKey(keyCode) -> {
                // 스페이스, 엔터, 숫자, 방향키 등: 조합을 끝내고 원래 처리에 맡긴다
                finishComposing()
            }
        }
        return super.onKeyPreIme(keyCode, event)
    }

    /** 조합 중인 글자를 확정한다. */
    fun finishComposing() {
        if (!composer.isComposing) return
        applyComposition(composer.commit())
    }

    /** 조합 영역을 '확정 문자열 + 현재 조합 글자' 로 바꾼다. */
    private fun applyComposition(committed: String) {
        val editable = text ?: return
        updating = true
        try {
            if (composeStart < 0) {
                val selStart = selectionStart.coerceAtLeast(0)
                val selEnd = selectionEnd.coerceAtLeast(0)
                composeStart = minOf(selStart, selEnd)
                editable.delete(composeStart, maxOf(selStart, selEnd))
            }
            val oldEnd = if (editable.getSpanStart(composingSpan) >= 0) editable.getSpanEnd(composingSpan) else composeStart
            editable.removeSpan(composingSpan)
            val composing = composer.composing
            editable.replace(composeStart, oldEnd, committed + composing)
            composeStart += committed.length
            val end = composeStart + composing.length
            if (composing.isNotEmpty()) {
                editable.setSpan(composingSpan, composeStart, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            } else {
                composeStart = -1
            }
            setSelection(end)
        } finally {
            updating = false
        }
    }

    override fun onSelectionChanged(selStart: Int, selEnd: Int) {
        super.onSelectionChanged(selStart, selEnd)
        if (!initialized) return
        // 커서를 다른 곳으로 옮기면 조합을 확정한다
        if (!updating && composeStart >= 0) {
            val end = composeStart + composer.composing.length
            if (selStart != end || selEnd != end) {
                updating = true
                text?.removeSpan(composingSpan)
                composer.commit()
                composeStart = -1
                updating = false
            }
        }
    }

    override fun onFocusChanged(focused: Boolean, direction: Int, previouslyFocusedRect: android.graphics.Rect?) {
        if (!focused && initialized) finishComposing()
        super.onFocusChanged(focused, direction, previouslyFocusedRect)
    }

    /** 외부에서 텍스트를 바꿀 때(지우기 등) 조합 상태도 초기화한다. */
    fun resetText(value: CharSequence) {
        composer.commit()
        composeStart = -1
        setText(value)
        setSelection(value.length)
    }

    private fun isToggleKey(event: KeyEvent): Boolean = when (event.keyCode) {
        KeyEvent.KEYCODE_LANGUAGE_SWITCH, KeyEvent.KEYCODE_KANA, KeyEvent.KEYCODE_EISU,
        KeyEvent.KEYCODE_ALT_RIGHT -> true
        KeyEvent.KEYCODE_SPACE -> event.isShiftPressed
        else -> false
    }

    companion object {
        private const val CTRL_ALT_META = KeyEvent.META_CTRL_ON or KeyEvent.META_ALT_ON or KeyEvent.META_META_ON

        fun isFromPhysicalKeyboard(event: KeyEvent): Boolean {
            val device = event.device ?: return false
            return !device.isVirtual && device.keyboardType == InputDevice.KEYBOARD_TYPE_ALPHABETIC
        }
    }
}
