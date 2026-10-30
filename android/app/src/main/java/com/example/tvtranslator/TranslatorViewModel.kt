package com.example.tvtranslator

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/** 모델 로딩과 번역을 전용 스레드 하나에서 수행한다 (TFLite Interpreter 는 thread-safe 하지 않음). */
class TranslatorViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface ModelState {
        data object Loading : ModelState
        data object Ready : ModelState
        data class Failed(val message: String) : ModelState
    }

    data class Result(val text: String, val target: Translator.Target, val millis: Long)

    private val executor = Executors.newSingleThreadExecutor()
    private val dispatcher = executor.asCoroutineDispatcher()
    private var translator: Translator? = null

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)
    val modelState: StateFlow<ModelState> = _modelState

    init {
        viewModelScope.launch(dispatcher) {
            _modelState.value = try {
                val t = Translator.create(getApplication())
                t.translate("hello", Translator.Target.KO)  // warm-up (첫 추론의 메모리 할당)
                translator = t
                ModelState.Ready
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                ModelState.Failed(e.toString())
            }
        }
    }

    suspend fun translate(text: String, target: Translator.Target): Result = withContext(dispatcher) {
        val t = checkNotNull(translator) { "model not loaded" }
        val start = SystemClock.elapsedRealtime()
        val out = t.translate(text, target)
        Result(out, target, SystemClock.elapsedRealtime() - start)
    }

    override fun onCleared() {
        executor.execute { translator?.close() }
        executor.shutdown()
    }

    private companion object {
        const val TAG = "TranslatorViewModel"
    }
}
