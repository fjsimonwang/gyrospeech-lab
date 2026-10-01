package com.research.gyrospeech

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var recorder: SensorRecorder
    private lateinit var genderClf: Classifier
    private lateinit var speakerClf: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        recorder = SensorRecorder(this)
        genderClf = Classifier(this, "gender_en").apply { ensureLabel("Male"); ensureLabel("Female") }
        speakerClf = Classifier(this, "speaker_en").apply {
            ensureLabel("Speaker A"); ensureLabel("Speaker B"); ensureLabel("Speaker C")
        }
        setContent { App(recorder, genderClf, speakerClf) }
    }

    override fun onResume() { super.onResume(); recorder.start(recorder.currentKind) }
    override fun onPause() { super.onPause(); recorder.stop(keepFile = true) }
}

private val bg = Color(0xFF0B0B12)
private val panel = Color(0xFF15151F)
private val accent = Color(0xFF5EE6C4)
private val accent2 = Color(0xFF7C9CFF)
private val warn = Color(0xFFF2C57C)
private val textDim = Color(0xFF9AA0B4)

private enum class Mode(val label: String) { GENDER("Gender"), SPEAKER("Speaker") }

@Composable
fun App(recorder: SensorRecorder, genderClf: Classifier, speakerClf: Classifier) {
    val lifecycleOwner = LocalLifecycleOwner.current
    var kind by remember { mutableStateOf(recorder.currentKind) }
    var recording by remember { mutableStateOf(false) }
    var statusMsg by remember { mutableStateOf("") }

    // 可视化状态
    val wave = remember { FloatArray(recorder.bufferSize) }
    var waveState by remember { mutableStateOf(FloatArray(recorder.bufferSize)) }
    var spectrum by remember { mutableStateOf(FloatArray(128)) }
    var hz by remember { mutableStateOf(0f) }
    var count by remember { mutableStateOf(0L) }
    var energy by remember { mutableStateOf(0f) }
    var pitchHz by remember { mutableStateOf(0f) }

    // 分类状态
    var mode by remember { mutableStateOf(Mode.GENDER) }
    val clf = if (mode == Mode.GENDER) genderClf else speakerClf
    var enrolling by remember { mutableStateOf<String?>(null) }
    var enrollProgress by remember { mutableStateOf(0) }
    val enrollTarget = 40 // ~4s @ 10Hz
    var prediction by remember { mutableStateOf<Classifier.Prediction?>(null) }
    var uiTick by remember { mutableStateOf(0) } // 触发样本数刷新

    // M4 状态
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var m4Busy by remember { mutableStateOf(false) }
    var m4Status by remember { mutableStateOf("") }
    var lastAccel by remember { mutableStateOf<FloatArray?>(null) }
    var lastWav by remember { mutableStateOf<File?>(null) }

    val voiceGate = 0.004f // 有声门限（gyro/accel 幅度 RMS）

    DisposableEffect(lifecycleOwner) {
        val obs = LifecycleEventObserver { _, e -> if (e == Lifecycle.Event.ON_STOP) recorder.stop(keepFile = true) }
        lifecycleOwner.lifecycle.addObserver(obs)
        onDispose { lifecycleOwner.lifecycle.removeObserver(obs) }
    }

    LaunchedEffect(kind) {
        recorder.start(kind)
        while (true) {
            recorder.snapshot(wave)
            waveState = wave.copyOf()
            val fftIn = FloatArray(256)
            val start = wave.size - 256
            for (i in 0 until 256) fftIn[i] = wave[start + i]
            spectrum = Fft.magnitudeSpectrum(fftIn)
            hz = recorder.measuredHz
            count = recorder.sampleCount
            energy = FeatureExtractor.energy(fftIn)
            pitchHz = FeatureExtractor.dominantHz(fftIn, hz)

            val voiced = energy > voiceGate
            val feat = FeatureExtractor.extract(fftIn, hz)

            val target = enrolling
            if (target != null) {
                if (voiced) {
                    clf.addSample(target, feat)
                    enrollProgress++
                    if (enrollProgress >= enrollTarget) {
                        clf.train()
                        enrolling = null
                        enrollProgress = 0
                        uiTick++
                        statusMsg = "Enrolled \"$target\" — ${clf.sampleCount(target)} samples"
                    }
                }
                prediction = null
            } else if (clf.isReady() && voiced) {
                prediction = clf.predict(feat)
            } else if (!voiced) {
                // 静音时不刷新预测，避免对噪声瞎猜
            }
            delay(100)
        }
    }

    MaterialTheme {
        Column(
            Modifier.fillMaxSize().background(bg).verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text("GyroSpeech Lab", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold)
            DisclaimerBanner()

            // 传感器切换
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SensorRecorder.Kind.values().forEach { k ->
                    val enabled = recorder.hasSensor(k)
                    Pill(k.label, selected = k == kind, enabled = enabled, modifier = Modifier.weight(1f)) { if (enabled) kind = k }
                }
            }

            MetricRow(hz, count, pitchHz)

            SectionCard("3-axis magnitude |a| (time domain)") { WaveformCanvas(waveState) }
            SectionCard("FFT magnitude spectrum — usable band ≈ 0…${(hz / 2).toInt()} Hz") { SpectrumCanvas(spectrum) }

            // ===== M2 分类 =====
            ClassifierCard(
                mode = mode,
                onModeChange = { mode = it; enrolling = null; enrollProgress = 0; prediction = null },
                clf = clf,
                enrolling = enrolling,
                enrollProgress = enrollProgress,
                enrollTarget = enrollTarget,
                prediction = prediction,
                energy = energy,
                voiceGate = voiceGate,
                uiTick = uiTick,
                onEnroll = { label -> enrolling = label; enrollProgress = 0; statusMsg = "Enrolling \"$label\" — keep speaking toward the desk…" },
                onClearAll = { clf.clearAll(); prediction = null; uiTick++; statusMsg = "Current model cleared" }
            )

            // ===== M4 加速度耦合 & 重建 =====
            M4Card(
                busy = m4Busy,
                status = m4Status,
                accelCount = lastAccel?.size ?: 0,
                hasWav = lastWav != null,
                onPlayProbe = {
                    scope.launch {
                        m4Busy = true; lastWav = null
                        kind = SensorRecorder.Kind.ACCEL
                        delay(500)
                        recorder.startCapture()
                        m4Status = "Playing probe tone & capturing 5s… (phone on desk, don't block the speaker)"
                        withContext(Dispatchers.Default) { AudioLab.playProbe(context, 5000) }
                        val a = recorder.stopCapture()
                        lastAccel = a
                        m4Status = "Captured ${a.size} pts @ ${"%.0f".format(recorder.measuredHz)}Hz. Tap \"Reconstruct\""
                        m4Busy = false
                    }
                },
                onCaptureOnly = {
                    scope.launch {
                        m4Busy = true; lastWav = null
                        kind = SensorRecorder.Kind.ACCEL
                        delay(500)
                        recorder.startCapture()
                        for (s in 5 downTo 1) { m4Status = "Capturing… play audio out loud (call/media). ${s}s left"; delay(1000) }
                        val a = recorder.stopCapture()
                        lastAccel = a
                        m4Status = "Captured ${a.size} pts @ ${"%.0f".format(recorder.measuredHz)}Hz. Tap \"Reconstruct\""
                        m4Busy = false
                    }
                },
                onReconstruct = {
                    val a = lastAccel
                    if (a == null || a.size < 32) { m4Status = "Capture first"; }
                    else scope.launch {
                        m4Busy = true; m4Status = "Reconstructing to audible WAV…"
                        val dir = context.getExternalFilesDir(null) ?: context.filesDir
                        val f = File(dir, "recon_${System.currentTimeMillis()}.wav")
                        val dur = withContext(Dispatchers.Default) { AudioLab.reconstructToWav(a, recorder.measuredHz, f) }
                        lastWav = f
                        m4Status = "Saved ${f.name} (${"%.1f".format(dur)}s) → tap \"Play\""
                        m4Busy = false
                    }
                },
                onPlayWav = { lastWav?.let { AudioLab.play(it) } }
            )

            // 录制 CSV
            Button(
                onClick = {
                    if (!recording) { val f = recorder.startRecording(); recording = true; statusMsg = "Recording → ${f.name}" }
                    else { recorder.stopRecording(); recording = false; statusMsg = "Saved: ${recorder.lastFile?.absolutePath ?: "-"}" }
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (recording) Color(0xFFE0596B) else panel,
                    contentColor = if (recording) Color.White else textDim
                ),
                shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()
            ) { Text(if (recording) "■ Stop & save CSV" else "● Record raw data to CSV") }

            if (statusMsg.isNotEmpty())
                Text(statusMsg, color = textDim, fontSize = 12.sp, fontFamily = FontFamily.Monospace, lineHeight = 16.sp)

            Text(
                "How to use: Enroll each class for a few seconds (that person keeps speaking toward the desk); "
                    + "with ≥2 classes it classifies live. The gyroscope only keeps content <${(hz / 2).toInt()}Hz — "
                    + "gender/speaker are separable, but verbatim transcription is not possible.",
                color = textDim, fontSize = 11.sp, lineHeight = 16.sp
            )
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ClassifierCard(
    mode: Mode,
    onModeChange: (Mode) -> Unit,
    clf: Classifier,
    enrolling: String?,
    enrollProgress: Int,
    enrollTarget: Int,
    prediction: Classifier.Prediction?,
    energy: Float,
    voiceGate: Float,
    uiTick: Int,
    onEnroll: (String) -> Unit,
    onClearAll: () -> Unit
) {
    Column(
        Modifier.fillMaxWidth().background(panel, RoundedCornerShape(12.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text("On-device classification (nearest-centroid · trained locally)", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Mode.values().forEach { m ->
                Pill(m.label, selected = m == mode, enabled = true, modifier = Modifier.weight(1f)) { onModeChange(m) }
            }
        }

        val voiced = energy > voiceGate
        Text(
            "Signal energy ${"%.4f".format(energy)} ${if (voiced) "· voiced ●" else "· silent (enroll/classify only while speaking)"}",
            color = if (voiced) accent else textDim, fontSize = 11.sp, fontFamily = FontFamily.Monospace
        )

        // 每个类别一行：名称 + 样本数 + 录入按钮
        key(uiTick) {
            clf.labels().forEach { label ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Column(Modifier.weight(1f)) {
                        Text(label, color = Color.White, fontSize = 13.sp)
                        Text("${clf.sampleCount(label)} samples", color = textDim, fontSize = 11.sp)
                    }
                    val isThis = enrolling == label
                    Button(
                        onClick = { if (enrolling == null) onEnroll(label) },
                        enabled = enrolling == null,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isThis) warn else accent2,
                            contentColor = bg,
                            disabledContainerColor = Color(0xFF2A2C38),
                            disabledContentColor = textDim
                        ),
                        shape = RoundedCornerShape(9.dp)
                    ) {
                        Text(if (isThis) "Enrolling ${enrollProgress * 100 / enrollTarget}%" else "Enroll", fontSize = 12.sp)
                    }
                }
            }
        }

        // Prediction
        Divider(color = Color(0xFF262838))
        if (prediction == null) {
            Text(
                if (clf.readyClasses() < 2) "Enroll at least 2 classes to start" else "Speak to see the prediction…",
                color = textDim, fontSize = 12.sp
            )
        } else {
            Text(
                "Prediction: ${prediction.best}   confidence ${"%.0f".format(prediction.confidence * 100)}%",
                color = accent, fontSize = 15.sp, fontWeight = FontWeight.Bold
            )
            prediction.probs.forEach { (label, p) -> ProbBar(label, p) }
        }

        TextButton(onClick = onClearAll) { Text("Clear current model", color = Color(0xFFE0596B), fontSize = 12.sp) }
    }
}

@Composable
private fun ProbBar(label: String, p: Float) {
    Column(Modifier.padding(vertical = 2.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, color = textDim, fontSize = 11.sp)
            Text("${"%.0f".format(p * 100)}%", color = textDim, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
        }
        Box(Modifier.fillMaxWidth().height(6.dp).background(Color(0xFF23252F), RoundedCornerShape(3.dp))) {
            Box(
                Modifier.fillMaxWidth(p.coerceIn(0f, 1f)).height(6.dp)
                    .background(accent, RoundedCornerShape(3.dp))
            )
        }
    }
}

@Composable
private fun Pill(text: String, selected: Boolean, enabled: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick, enabled = enabled,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) accent else Color(0xFF1E2029),
            contentColor = if (selected) bg else textDim,
            disabledContainerColor = Color(0xFF1A1C24), disabledContentColor = Color(0xFF55596B)
        ),
        shape = RoundedCornerShape(10.dp), modifier = modifier
    ) { Text(text, fontSize = 13.sp) }
}

@Composable
private fun M4Card(
    busy: Boolean,
    status: String,
    accelCount: Int,
    hasWav: Boolean,
    onPlayProbe: () -> Unit,
    onCaptureOnly: () -> Unit,
    onReconstruct: () -> Unit,
    onPlayWav: () -> Unit
) {
    Column(
        Modifier.fillMaxWidth().background(panel, RoundedCornerShape(12.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text("M4 · Same-phone loudspeaker coupling → audible reconstruction (accelerometer)", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        Text(
            "Speaker vibration couples into the accelerometer through the chassis. Reconstructs audio at prosody/pitch level (DSP baseline, not verbatim transcription).",
            color = textDim, fontSize = 11.sp, lineHeight = 16.sp
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onPlayProbe, enabled = !busy,
                colors = ButtonDefaults.buttonColors(containerColor = accent2, contentColor = bg,
                    disabledContainerColor = Color(0xFF2A2C38), disabledContentColor = textDim),
                shape = RoundedCornerShape(9.dp), modifier = Modifier.weight(1f)
            ) { Text("▶ Probe + capture 5s", fontSize = 12.sp) }
            Button(
                onClick = onCaptureOnly, enabled = !busy,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E2029), contentColor = textDim,
                    disabledContainerColor = Color(0xFF1A1C24), disabledContentColor = Color(0xFF55596B)),
                shape = RoundedCornerShape(9.dp), modifier = Modifier.weight(1f)
            ) { Text("● Capture only 5s", fontSize = 12.sp) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onReconstruct, enabled = !busy && accelCount >= 32,
                colors = ButtonDefaults.buttonColors(containerColor = accent, contentColor = bg,
                    disabledContainerColor = Color(0xFF2A2C38), disabledContentColor = textDim),
                shape = RoundedCornerShape(9.dp), modifier = Modifier.weight(1f)
            ) { Text("🔊 Reconstruct WAV", fontSize = 12.sp) }
            Button(
                onClick = onPlayWav, enabled = !busy && hasWav,
                colors = ButtonDefaults.buttonColors(containerColor = warn, contentColor = bg,
                    disabledContainerColor = Color(0xFF2A2C38), disabledContentColor = textDim),
                shape = RoundedCornerShape(9.dp), modifier = Modifier.weight(1f)
            ) { Text("▶ Play", fontSize = 12.sp) }
        }
        if (accelCount > 0) Text("Captured $accelCount pts", color = textDim, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
        if (status.isNotEmpty()) Text(status, color = if (busy) warn else accent, fontSize = 12.sp, lineHeight = 16.sp)
    }
}

@Composable
private fun DisclaimerBanner() {
    Box(Modifier.fillMaxWidth().background(Color(0xFF2A1E12), RoundedCornerShape(10.dp)).padding(12.dp)) {
        Text(
            "⚠ Research/education use only. Use on your own device or with explicit consent. Capturing others' speech without consent may be illegal.",
            color = warn, fontSize = 12.sp, lineHeight = 17.sp
        )
    }
}

@Composable
private fun MetricRow(hz: Float, count: Long, pitchHz: Float) {
    Row(
        Modifier.fillMaxWidth().background(panel, RoundedCornerShape(12.dp)).padding(vertical = 14.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        Metric("Sample rate", "%.1f Hz".format(hz), accent)
        Metric("Nyquist", "%.0f Hz".format(hz / 2), accent2)
        Metric("Pitch≈F0", "%.0f Hz".format(pitchHz), warn)
        Metric("Samples", "$count", Color.White)
    }
}

@Composable
private fun Metric(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontSize = 18.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
        Text(label, color = textDim, fontSize = 11.sp)
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().background(panel, RoundedCornerShape(12.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(title, color = textDim, fontSize = 12.sp)
        content()
    }
}

@Composable
private fun WaveformCanvas(wave: FloatArray) {
    Canvas(Modifier.fillMaxWidth().height(110.dp)) {
        if (wave.isEmpty()) return@Canvas
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
        for (v in wave) { if (v < mn) mn = v; if (v > mx) mx = v }
        val range = (mx - mn).coerceAtLeast(1e-4f)
        val path = Path()
        for (i in wave.indices) {
            val x = size.width * i / (wave.size - 1)
            val y = size.height * (1f - (wave[i] - mn) / range)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, accent, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f))
    }
}

@Composable
private fun SpectrumCanvas(spectrum: FloatArray) {
    Canvas(Modifier.fillMaxWidth().height(130.dp)) {
        if (spectrum.isEmpty()) return@Canvas
        var mx = 1e-6f
        for (v in spectrum) if (v > mx) mx = v
        val barW = size.width / spectrum.size
        for (i in spectrum.indices) {
            val h = size.height * (spectrum[i] / mx)
            drawRect(
                color = androidx.compose.ui.graphics.lerp(accent2, accent, i.toFloat() / spectrum.size),
                topLeft = androidx.compose.ui.geometry.Offset(barW * i, size.height - h),
                size = androidx.compose.ui.geometry.Size(barW * 0.85f, h)
            )
        }
    }
}
