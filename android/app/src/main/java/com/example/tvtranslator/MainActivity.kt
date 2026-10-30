package com.example.tvtranslator

import android.os.Bundle
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private enum class Direction { AUTO, KO_TO_EN, EN_TO_KO }

    private val viewModel: TranslatorViewModel by viewModels()

    private lateinit var inputText: HangulEditText
    private lateinit var outputText: TextView
    private lateinit var outputLabel: TextView
    private lateinit var statusText: TextView
    private lateinit var directionButton: Button
    private lateinit var keyboardButton: Button
    private lateinit var translateButton: Button

    private var direction = Direction.AUTO
    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputText = findViewById(R.id.input_text)
        outputText = findViewById(R.id.output_text)
        outputLabel = findViewById(R.id.output_label)
        statusText = findViewById(R.id.status_text)
        directionButton = findViewById(R.id.direction_button)
        keyboardButton = findViewById(R.id.keyboard_button)
        translateButton = findViewById(R.id.translate_button)

        direction = savedInstanceState?.getString(KEY_DIRECTION)?.let { Direction.valueOf(it) } ?: Direction.AUTO
        inputText.hangulMode = savedInstanceState?.getBoolean(KEY_HANGUL_MODE) ?: false

        directionButton.setOnClickListener {
            direction = Direction.entries[(direction.ordinal + 1) % Direction.entries.size]
            updateDirectionViews()
        }
        keyboardButton.setOnClickListener { inputText.hangulMode = !inputText.hangulMode }
        inputText.onHangulModeChanged = { updateKeyboardButton() }
        translateButton.setOnClickListener { translate() }
        findViewById<Button>(R.id.clear_button).setOnClickListener {
            inputText.resetText("")
            outputText.text = ""
            inputText.requestFocus()
        }
        findViewById<Button>(R.id.use_result_button).setOnClickListener {
            if (outputText.text.isNotEmpty()) {
                inputText.resetText(outputText.text)
                outputText.text = ""
                if (direction != Direction.AUTO) {
                    direction = if (direction == Direction.KO_TO_EN) Direction.EN_TO_KO else Direction.KO_TO_EN
                    updateDirectionViews()
                }
            }
        }

        // 한 줄 입력(inputType=text)이어야 IME 에 '완료' 버튼이 나온다. 대신 화면에서는 줄바꿈해서 보여 준다.
        inputText.setHorizontallyScrolling(false)
        inputText.maxLines = Int.MAX_VALUE

        // 화상 키보드의 '완료' 또는 물리 키보드 Enter → 번역
        inputText.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                translate()
                hideKeyboard()
                true
            } else false
        }
        inputText.setOnKeyListener { _, keyCode, event ->
            if (keyCode == KeyEvent.KEYCODE_ENTER || keyCode == KeyEvent.KEYCODE_NUMPAD_ENTER) {
                if (event.action == KeyEvent.ACTION_DOWN) translate()
                true
            } else false
        }

        updateDirectionViews()
        updateKeyboardButton()

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.modelState.collect { state ->
                    when (state) {
                        TranslatorViewModel.ModelState.Loading -> setStatus(getString(R.string.status_loading))
                        TranslatorViewModel.ModelState.Ready -> setStatus(getString(R.string.status_ready))
                        is TranslatorViewModel.ModelState.Failed -> setStatus(getString(R.string.status_model_missing, state.message))
                    }
                    translateButton.isEnabled = state == TranslatorViewModel.ModelState.Ready
                }
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(KEY_DIRECTION, direction.name)
        outState.putBoolean(KEY_HANGUL_MODE, inputText.hangulMode)
    }

    /** 입력창 밖에 포커스가 있을 때 물리 키보드로 글자를 치면 입력창으로 보낸다. */
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (!inputText.hasFocus() && event.action == KeyEvent.ACTION_DOWN && event.isPrintingKey &&
            event.keyCode != KeyEvent.KEYCODE_SPACE && HangulEditText.isFromPhysicalKeyboard(event)
        ) {
            inputText.requestFocus()
            if (!inputText.onKeyPreIme(event.keyCode, event)) inputText.dispatchKeyEvent(event)
            return true
        }
        return super.dispatchKeyEvent(event)
    }

    private fun translate() {
        if (busy || viewModel.modelState.value != TranslatorViewModel.ModelState.Ready) return
        inputText.finishComposing()
        val text = inputText.text?.toString().orEmpty().trim()
        if (text.isEmpty()) return
        val target = when (direction) {
            Direction.AUTO -> Translator.detectTarget(text)
            Direction.KO_TO_EN -> Translator.Target.EN
            Direction.EN_TO_KO -> Translator.Target.KO
        }
        busy = true
        setStatus(getString(R.string.status_translating))
        lifecycleScope.launch {
            try {
                val result = viewModel.translate(text, target)
                outputText.text = result.text
                outputLabel.text = getString(R.string.output_label, targetName(result.target))
                setStatus(getString(R.string.status_done, result.millis / 1000.0))
            } catch (e: Exception) {
                setStatus(getString(R.string.status_error, e.toString()))
            } finally {
                busy = false
            }
        }
    }

    private fun updateDirectionViews() {
        directionButton.setText(
            when (direction) {
                Direction.AUTO -> R.string.direction_auto
                Direction.KO_TO_EN -> R.string.direction_ko_en
                Direction.EN_TO_KO -> R.string.direction_en_ko
            }
        )
        outputLabel.text = getString(
            R.string.output_label,
            when (direction) {
                Direction.AUTO -> getString(R.string.lang_auto)
                Direction.KO_TO_EN -> targetName(Translator.Target.EN)
                Direction.EN_TO_KO -> targetName(Translator.Target.KO)
            },
        )
    }

    private fun updateKeyboardButton() {
        keyboardButton.setText(if (inputText.hangulMode) R.string.keyboard_hangul else R.string.keyboard_english)
    }

    private fun targetName(target: Translator.Target) =
        getString(if (target == Translator.Target.KO) R.string.lang_ko else R.string.lang_en)

    private fun setStatus(message: String) {
        statusText.text = message
    }

    private fun hideKeyboard() {
        getSystemService(InputMethodManager::class.java).hideSoftInputFromWindow(inputText.windowToken, 0)
    }

    private companion object {
        const val KEY_DIRECTION = "direction"
        const val KEY_HANGUL_MODE = "hangul_mode"
    }
}
