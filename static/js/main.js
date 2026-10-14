const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const selectImageButton = document.getElementById('select-image-button');
const resultSection = document.getElementById('result-section');
const ACCEPTED_FORMATS = ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'];

let selectedFile = null;

let selectedWindowPreset = 'auto';
let volumeInfo = null;

const huPresetButtons = document.querySelectorAll('.hu-preset-btn');
const sliceSliderWrapper = document.getElementById('slice-slider-wrapper');
const sliceRange = document.getElementById('slice-range');
const sliceCounter = document.getElementById('slice-counter');
const volumeIndicator = document.getElementById('volume-indicator');

huPresetButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    huPresetButtons.forEach((b) => {
      b.classList.remove('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
      b.classList.add('border-slate-800', 'bg-slate-900', 'text-slate-300');
    });
    btn.classList.add('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
    btn.classList.remove('border-slate-800', 'bg-slate-900', 'text-slate-300');
    selectedWindowPreset = btn.dataset.preset || 'auto';
  });
});
let selectedModality = 'general';
const modalityButtons = document.querySelectorAll('.modality-btn');

modalityButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    modalityButtons.forEach((b) => {
      b.classList.remove('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
      b.classList.add('border-slate-800', 'bg-slate-900', 'text-slate-300');
    });
    btn.classList.add('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
    btn.classList.remove('border-slate-800', 'bg-slate-900', 'text-slate-300');
    selectedModality = btn.dataset.modality || 'general';
  });
});

let selectedScale = 8;
const scaleButtons = document.querySelectorAll('.scale-btn');

scaleButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    scaleButtons.forEach((b) => {
      b.classList.remove('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
      b.classList.add('border-slate-800', 'bg-slate-900', 'text-slate-300');
    });
    btn.classList.add('active', 'border-cyan-400', 'bg-cyan-500/20', 'text-cyan-200');
    btn.classList.remove('border-slate-800', 'bg-slate-900', 'text-slate-300');
    selectedScale = parseInt(btn.dataset.scale, 10) || 8;
  });
});

if (sliceRange && sliceCounter) {
  sliceRange.addEventListener('input', () => {
    const val = parseInt(sliceRange.value, 10);
    const max = parseInt(sliceRange.max, 10) + 1;
    sliceCounter.textContent = `Slice ${val + 1} / ${max}`;
  });
}

async function inspectVolumeFile(file) {
  const lowerName = file.name.toLowerCase();
  const isVolCandidate = lowerName.endsWith('.dcm') || lowerName.endsWith('.nii') || lowerName.endsWith('.nii.gz');
  if (!isVolCandidate) {
    if (sliceSliderWrapper) sliceSliderWrapper.classList.add('hidden');
    if (volumeIndicator) volumeIndicator.textContent = '2D Scan';
    volumeInfo = null;
    return;
  }

  try {
    const checkForm = new FormData();
    checkForm.append('file', file);
    const res = await fetch('/api/volume/inspect', {
      method: 'POST',
      body: checkForm,
    });
    if (res.ok) {
      const data = await res.json();
      volumeInfo = data;
      if (data.is_volume && data.total_slices > 1) {
        if (sliceSliderWrapper) sliceSliderWrapper.classList.remove('hidden');
        if (sliceRange) {
          sliceRange.min = '0';
          sliceRange.max = (data.total_slices - 1).toString();
          sliceRange.value = data.default_slice.toString();
        }
        if (sliceCounter) {
          sliceCounter.textContent = `Slice ${data.default_slice + 1} / ${data.total_slices}`;
        }
        if (volumeIndicator) {
          volumeIndicator.textContent = `3D Volume (${data.total_slices} slices)`;
          volumeIndicator.classList.add('text-cyan-400');
        }
      } else {
        if (sliceSliderWrapper) sliceSliderWrapper.classList.add('hidden');
        if (volumeIndicator) volumeIndicator.textContent = '2D Medical Scan';
      }
    }
  } catch (err) {
    console.warn('Volume inspection failed:', err);
  }
}

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => fileInput.click());
  if (selectImageButton) {
    selectImageButton.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }
  fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
      dropZone.innerHTML = `<p class="text-slate-200">Selected: ${selectedFile.name}</p>`;
      inspectVolumeFile(selectedFile);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    selectedFile = e.dataTransfer.files[0];
    if (selectedFile) {
      dropZone.innerHTML = `<p class="text-slate-200">Selected: ${selectedFile.name}</p>`;
      inspectVolumeFile(selectedFile);
    }
  });
}


if (uploadButton) {
  uploadButton.addEventListener('click', () => {
    if (!selectedFile) {
      resultSection.classList.remove('hidden');
      resultSection.innerHTML = `
        <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
          <p class="font-semibold">Please choose a file first.</p>
        </div>
      `;
      return;
    }
    handleFile(selectedFile);
  });
}

async function handleFile(file) {
  const lowerName = file.name.toLowerCase();
  const isAccepted = ACCEPTED_FORMATS.some((ext) => lowerName.endsWith(ext));
  if (!isAccepted) {
    resultSection.classList.remove('hidden');
    resultSection.innerHTML = `
      <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
        <p class="font-semibold">Unsupported file type.</p>
        <p class="mt-2 text-sm text-slate-400">Please choose a PNG, JPG, JPEG, DICOM, or NIfTI file.</p>
      </div>
    `;
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('window_preset', selectedWindowPreset);
  formData.append('scale', selectedScale);
  formData.append('modality', selectedModality);
  if (sliceSliderWrapper && !sliceSliderWrapper.classList.contains('hidden') && sliceRange) {
    formData.append('slice_index', sliceRange.value);
  }

  // Progress Card DOM references
  const progressCard = document.getElementById('processing-progress-card');
  const progressStepText = document.getElementById('progress-step-text');
  const progressPctText = document.getElementById('progress-pct-text');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressScaleBadge = document.getElementById('progress-scale-badge');
  const progressHipaaBadge = document.getElementById('progress-hipaa-badge');

  if (progressCard) {
    resultSection.classList.add('hidden');
    progressCard.classList.remove('hidden');
    if (progressStepText) progressStepText.textContent = 'Uploading scan to neural processing pipeline...';
    if (progressPctText) progressPctText.textContent = '5%';
    if (progressBarFill) progressBarFill.style.width = '5%';
    if (progressScaleBadge) progressScaleBadge.textContent = `QGAN ${selectedScale}x`;
    if (progressHipaaBadge) {
      const isDcm = lowerName.endsWith('.dcm') || lowerName.endsWith('.dicom');
      progressHipaaBadge.innerHTML = isDcm
        ? '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> HIPAA Safe Harbor PHI Sanitization Active'
        : '<span class="w-1.5 h-1.5 rounded-full bg-cyan-400"></span> Standard Scan Ingestion';
    }
  }

  if (uploadButton) {
    uploadButton.disabled = true;
    uploadButton.classList.add('opacity-50', 'cursor-not-allowed');
  }

  try {
    const uploadRes = await fetch('/upload/async', {
      method: 'POST',
      headers: { Accept: 'application/json' },
      body: formData,
    });
    const uploadData = await uploadRes.json();

    if (!uploadRes.ok || !uploadData.job_id) {
      throw new Error(uploadData.error || 'Failed to dispatch async enhancement job.');
    }

    const jobId = uploadData.job_id;

    // Poll job status until complete or error
    const pollInterval = setInterval(async () => {
      try {
        const jobRes = await fetch(`/api/jobs/${jobId}`);
        const job = await jobRes.json();

        if (job.progress !== undefined && progressBarFill && progressPctText && progressStepText) {
          const pct = Math.max(5, Math.min(100, job.progress));
          progressBarFill.style.width = `${pct}%`;
          progressPctText.textContent = `${pct}%`;
          if (job.step) progressStepText.textContent = job.step;
        }

        if (job.status === 'completed' && job.data) {
          clearInterval(pollInterval);
          if (progressCard) progressCard.classList.add('hidden');
          if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.classList.remove('opacity-50', 'cursor-not-allowed');
          }

          resultSection.classList.remove('hidden');
          const data = job.data;
          renderComparisonViewer(
            resultSection,
            data.original,
            data.result,
            data.message,
            `✅ QGAN ${data.scale_factor || 8}x Enhanced Result`,
            data.metrics,
            data.window_preset,
            data.slice_index,
            data.report_url,
            data.scale_factor || 8,
            data.dicom_url,
            data.deidentified,
            data.modality || 'general'
          );
        } else if (job.status === 'error') {
          clearInterval(pollInterval);
          if (progressCard) progressCard.classList.add('hidden');
          if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.classList.remove('opacity-50', 'cursor-not-allowed');
          }

          resultSection.classList.remove('hidden');
          resultSection.innerHTML = `
            <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
              <p class="font-semibold">Enhancement failed.</p>
              <p class="mt-2 text-sm text-slate-400">${job.error || 'An unexpected error occurred during processing.'}</p>
            </div>
          `;
        }
      } catch (pollErr) {
        console.warn('Poll error:', pollErr);
      }
    }, 350);

  } catch (error) {
    if (progressCard) progressCard.classList.add('hidden');
    if (uploadButton) {
      uploadButton.disabled = false;
      uploadButton.classList.remove('opacity-50', 'cursor-not-allowed');
    }
    resultSection.classList.remove('hidden');
    resultSection.innerHTML = `
      <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
        <p class="font-semibold">Upload failed.</p>
        <p class="mt-2 text-sm text-slate-400">${error.message || 'Please try again with a valid medical image file.'}</p>
      </div>
    `;
  }
}

/**
 * Creates and mounts an interactive Comparison Viewer component into targetElement
 */
function renderComparisonViewer(
  targetElement,
  originalUrl,
  resultUrl,
  message = '',
  title = '✅ QGAN Enhanced Result',
  metrics = null,
  windowPreset = null,
  sliceIndex = null,
  reportUrl = null,
  scaleFactor = 8,
  dicomUrl = null,
  deidentified = false,
  modality = 'general'
) {
  const viewerId = 'viewer_' + Math.random().toString(36).substr(2, 9);

  targetElement.innerHTML = `
    <div class="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 sm:p-8 shadow-2xl space-y-6" id="${viewerId}">
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 class="text-2xl sm:text-3xl font-semibold text-slate-100">${title}</h2>
          <div class="flex flex-wrap items-center gap-2 mt-1">
            <p class="text-sm text-cyan-400">Drag the center divider or switch view modes to inspect fine resolution details</p>
            ${deidentified ? `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-950 border border-emerald-700/60 text-emerald-300 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> HIPAA Safe Harbor PHI Sanitized</span>` : ''}
            ${modality && modality !== 'general' ? `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-violet-950 border border-violet-700/60 text-violet-300">Modality: ${modality.replace('_', ' ')}</span>` : ''}
            ${windowPreset && windowPreset !== 'auto' ? `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-indigo-950 border border-indigo-700/60 text-indigo-300">Preset: ${windowPreset}</span>` : ''}
            ${sliceIndex !== null && sliceIndex !== undefined && sliceIndex !== '' ? `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-950 border border-cyan-700/60 text-cyan-300">Slice #${parseInt(sliceIndex, 10) + 1}</span>` : ''}
          </div>
        </div>
        <div class="flex items-center gap-2 bg-slate-950 border border-slate-800 p-1.5 rounded-2xl">
          <button type="button" class="view-mode-btn active px-3.5 py-1.5 rounded-xl text-xs font-medium text-cyan-300 bg-slate-800 transition" data-mode="split">
            🔀 Split Slider
          </button>
          <button type="button" class="view-mode-btn px-3.5 py-1.5 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 transition" data-mode="side">
            🪟 Side by Side
          </button>
          <button type="button" class="view-mode-btn px-3.5 py-1.5 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 transition" data-mode="loupe">
            🔍 Magnifier
          </button>
        </div>
      </div>

      <!-- Split Slider View -->
      <div class="view-panel view-split block">
        <div class="compare-wrapper" id="${viewerId}_wrapper">
          <div class="compare-viewport" id="${viewerId}_viewport">
            <!-- Enhanced Image (Base Background) -->
            <img src="${resultUrl}" alt="Enhanced High-Resolution" class="compare-img compare-enhanced" id="${viewerId}_enhanced">
            <span class="compare-label compare-label-after">${scaleFactor || 8}x Enhanced</span>

            <!-- Original Image (Clipped Overlay) -->
            <div class="compare-overlay" id="${viewerId}_overlay">
              <img src="${originalUrl}" alt="Original Low-Resolution" class="compare-img compare-original" id="${viewerId}_original">
              <span class="compare-label compare-label-before">Original</span>
              <div class="compare-slider-handle" id="${viewerId}_handle" title="Drag to compare" tabindex="0">
                <span>&#8646;</span>
              </div>
            </div>

            <!-- Magnifier Lens -->
            <div class="magnifier-lens" id="${viewerId}_lens"></div>
          </div>
        </div>
        <div class="flex justify-between items-center text-xs text-slate-500 mt-3 px-2">
          <span>◂ Original (Low-Res)</span>
          <span class="text-slate-400">Click & drag divider bar</span>
          <span>QGAN Enhanced (High-Res) ▸</span>
        </div>
      </div>

      <!-- Side by Side View -->
      <div class="view-panel view-side hidden">
        <div class="grid gap-4 lg:grid-cols-2">
          <div class="rounded-2xl overflow-hidden border border-slate-800 bg-slate-950">
            <div class="px-4 py-3 bg-slate-900/80 text-sm font-medium text-slate-400 flex justify-between">
              <span>Original Low-Resolution</span>
              <span class="text-xs text-slate-500">1x Input</span>
            </div>
            <img src="${originalUrl}" alt="Original image" class="w-full h-80 sm:h-96 object-contain bg-black">
          </div>
          <div class="rounded-2xl overflow-hidden border border-slate-800 bg-slate-950">
            <div class="px-4 py-3 bg-slate-900/80 text-sm font-medium text-emerald-300 flex justify-between">
              <span>QGAN Super-Resolved</span>
              <span class="text-xs text-emerald-400">${scaleFactor || 8}x Enhanced</span>
            </div>
            <img src="${resultUrl}" alt="Enhanced image" class="w-full h-80 sm:h-96 object-contain bg-black">
          </div>
        </div>
      </div>

      <!-- Image Quality Assessment (IQA) Metrics Dashboard -->
      ${metrics ? `
      <div class="rounded-2xl border border-slate-800 bg-slate-950/90 p-4 sm:p-5 shadow-inner">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-3 pb-2 border-b border-slate-800/60">
          <div class="flex items-center gap-2">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h4 class="text-xs uppercase tracking-widest text-slate-200 font-semibold">Image Quality Assessment (IQA)</h4>
          </div>
          <span class="text-xs text-slate-400 font-mono">
            ${metrics.scale_factor ? `Scale: <strong class="text-cyan-300">${metrics.scale_factor}</strong>` : ''} 
            ${metrics.input_resolution ? `(${metrics.input_resolution} ➔ ${metrics.output_resolution})` : ''}
          </span>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-cyan-500/30 transition">
            <div class="text-xs text-slate-400">PSNR Fidelity</div>
            <div class="text-lg sm:text-xl font-bold text-cyan-300 mt-1">${metrics.psnr !== undefined && metrics.psnr !== null && metrics.psnr !== '' ? metrics.psnr + ' dB' : '--'}</div>
            <div class="text-[11px] text-slate-500 mt-0.5">Peak Signal-to-Noise</div>
          </div>
          <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-emerald-500/30 transition">
            <div class="text-xs text-slate-400">SSIM Index</div>
            <div class="text-lg sm:text-xl font-bold text-emerald-300 mt-1">${metrics.ssim !== undefined && metrics.ssim !== null && metrics.ssim !== '' ? metrics.ssim : '--'}</div>
            <div class="text-[11px] text-slate-500 mt-0.5">Structural Similarity (0-1)</div>
          </div>
          <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-indigo-500/30 transition">
            <div class="text-xs text-slate-400">Edge Sharpness</div>
            <div class="text-lg sm:text-xl font-bold text-indigo-300 mt-1">${metrics.sharpness !== undefined && metrics.sharpness !== null && metrics.sharpness !== '' ? metrics.sharpness : '--'}</div>
            <div class="text-[11px] text-slate-500 mt-0.5">Laplacian Variance</div>
          </div>
          <div class="rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-amber-500/30 transition">
            <div class="text-xs text-slate-400">Detail Gain</div>
            <div class="text-lg sm:text-xl font-bold text-amber-300 mt-1">${metrics.sharpness_gain_pct !== undefined ? (metrics.sharpness_gain_pct >= 0 ? '+' : '') + metrics.sharpness_gain_pct + '%' : '+8x Sharpness'}</div>
            <div class="text-[11px] text-slate-500 mt-0.5">Over Bicubic baseline</div>
          </div>
        </div>
      </div>
      ` : ''}

      <!-- Status & Download -->
      <div class="pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4">
        <p class="text-emerald-300 text-sm font-medium">${message}</p>
        <div class="flex flex-wrap items-center gap-3">
          ${dicomUrl ? `
          <a href="${dicomUrl}" download class="inline-flex items-center gap-1.5 rounded-full border border-indigo-400 bg-indigo-500/20 px-4 py-2.5 text-sm font-semibold text-indigo-200 hover:bg-indigo-500/30 transition shadow-sm">
            💾 Export .DCM
          </a>
          ` : ''}
          ${reportUrl ? `
          <a href="${reportUrl}" target="_blank" class="inline-flex items-center gap-1.5 rounded-full border border-cyan-400 bg-cyan-500/20 px-4 py-2.5 text-sm font-semibold text-cyan-200 hover:bg-cyan-500/30 transition shadow-sm">
            📄 Clinical Report
          </a>
          ` : ''}
          <a href="${resultUrl}" download class="inline-flex items-center gap-2 rounded-full bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400 shadow-lg shadow-cyan-500/20 transition">
            ⬇ Download ${scaleFactor || 8}x Result
          </a>
          <a href="/gallery" class="inline-flex items-center rounded-full border border-slate-700 bg-slate-800/60 px-4 py-2.5 text-sm text-slate-200 hover:border-slate-500 transition">
            View in Gallery
          </a>
        </div>
      </div>
    </div>
  `;

  initViewerLogic(viewerId, originalUrl, resultUrl);
}

/**
 * Initializes slider dragging, window resize synchronization, and magnifier lens logic
 */
function initViewerLogic(viewerId, originalUrl, resultUrl) {
  const container = document.getElementById(viewerId);
  if (!container) return;

  const viewport = document.getElementById(`${viewerId}_viewport`);
  const overlay = document.getElementById(`${viewerId}_overlay`);
  const handle = document.getElementById(`${viewerId}_handle`);
  const lens = document.getElementById(`${viewerId}_lens`);

  const modeButtons = container.querySelectorAll('.view-mode-btn');
  const viewSplit = container.querySelector('.view-split');
  const viewSide = container.querySelector('.view-side');

  let currentMode = 'split';
  let isDragging = false;

  // Sync width for CSS overlay calculation
  function syncViewportWidth() {
    if (viewport) {
      const w = viewport.clientWidth;
      viewport.style.setProperty('--viewport-width', `${w}px`);
    }
  }

  // Initial and delayed sync (to allow image render)
  syncViewportWidth();
  setTimeout(syncViewportWidth, 100);
  window.addEventListener('resize', syncViewportWidth);

  // Split Slider Dragging logic
  function setSliderPosition(xCoord) {
    const rect = viewport.getBoundingClientRect();
    let posX = xCoord - rect.left;
    if (posX < 0) posX = 0;
    if (posX > rect.width) posX = rect.width;
    const percentage = (posX / rect.width) * 100;
    const boundedPct = Math.max(2, Math.min(98, percentage));
    overlay.style.width = `${boundedPct}%`;
  }

  function onMouseDown(e) {
    if (currentMode === 'loupe') return;
    isDragging = true;
    setSliderPosition(e.clientX);
  }

  function onMouseMove(e) {
    if (isDragging && currentMode !== 'loupe') {
      setSliderPosition(e.clientX);
    }
    if (currentMode === 'loupe') {
      updateMagnifier(e);
    }
  }

  function onMouseUp() {
    isDragging = false;
  }

  function onTouchStart(e) {
    if (currentMode === 'loupe') return;
    isDragging = true;
    setSliderPosition(e.touches[0].clientX);
  }

  function onTouchMove(e) {
    if (isDragging && currentMode !== 'loupe') {
      setSliderPosition(e.touches[0].clientX);
    }
  }

  viewport.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);

  viewport.addEventListener('touchstart', onTouchStart, { passive: true });
  window.addEventListener('touchmove', onTouchMove, { passive: true });
  window.addEventListener('touchend', onMouseUp);

  // Keyboard accessibility on handle
  if (handle) {
    handle.addEventListener('keydown', (e) => {
      const currentPct = parseFloat(overlay.style.width) || 50;
      if (e.key === 'ArrowLeft') {
        overlay.style.width = `${Math.max(2, currentPct - 5)}%`;
      } else if (e.key === 'ArrowRight') {
        overlay.style.width = `${Math.min(98, currentPct + 5)}%`;
      }
    });
  }

  // Magnifier Lens logic
  const zoomFactor = 2.5;
  lens.style.backgroundImage = `url('${resultUrl}')`;

  function updateMagnifier(e) {
    const rect = viewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      lens.style.display = 'none';
      return;
    }

    lens.style.display = 'block';
    const lensRadius = 70; // 140px / 2
    lens.style.left = `${x - lensRadius}px`;
    lens.style.top = `${y - lensRadius}px`;

    lens.style.backgroundSize = `${rect.width * zoomFactor}px ${rect.height * zoomFactor}px`;
    const bgX = -(x * zoomFactor - lensRadius);
    const bgY = -(y * zoomFactor - lensRadius);
    lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
  }

  viewport.addEventListener('mouseleave', () => {
    lens.style.display = 'none';
  });

  // Mode Buttons handler
  modeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      modeButtons.forEach((b) => {
        b.classList.remove('active', 'text-cyan-300', 'bg-slate-800');
        b.classList.add('text-slate-400');
      });
      btn.classList.add('active', 'text-cyan-300', 'bg-slate-800');
      btn.classList.remove('text-slate-400');

      currentMode = btn.dataset.mode;

      if (currentMode === 'side') {
        viewSplit.classList.add('hidden');
        viewSide.classList.remove('hidden');
        lens.style.display = 'none';
      } else if (currentMode === 'loupe') {
        viewSide.classList.add('hidden');
        viewSplit.classList.remove('hidden');
        overlay.style.width = '0%'; // show full enhanced image underneath
        syncViewportWidth();
      } else {
        // split mode
        viewSide.classList.add('hidden');
        viewSplit.classList.remove('hidden');
        overlay.style.width = '50%';
        lens.style.display = 'none';
        syncViewportWidth();
      }
    });
  });
}

/**
 * Gallery Interactive Modal
 */
document.addEventListener('DOMContentLoaded', () => {
  const compareButtons = document.querySelectorAll('.btn-inspect-compare');
  if (!compareButtons.length) return;

  let modal = document.getElementById('gallery-compare-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'gallery-compare-modal';
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop hidden';
    modal.innerHTML = `
      <div class="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto rounded-3xl bg-slate-900 border border-slate-800 p-6 shadow-2xl">
        <button id="modal-close-btn" class="absolute top-5 right-5 text-slate-400 hover:text-white text-2xl font-bold bg-slate-800 w-10 h-10 rounded-full flex items-center justify-center transition">
          &times;
        </button>
        <div id="modal-viewer-container"></div>
      </div>
    `;
    document.body.appendChild(modal);

    const closeBtn = document.getElementById('modal-close-btn');
    closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  }

  const modalContainer = document.getElementById('modal-viewer-container');

  compareButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const original = btn.dataset.original;
      const result = btn.dataset.result;
      const title = btn.dataset.title || 'Interactive Scan Comparison';
      const psnr = btn.dataset.psnr ? parseFloat(btn.dataset.psnr) : null;
      const ssim = btn.dataset.ssim ? parseFloat(btn.dataset.ssim) : null;
      const sharpness = btn.dataset.sharpness ? parseFloat(btn.dataset.sharpness) : null;

      let metrics = null;
      if (psnr !== null || ssim !== null || sharpness !== null) {
        metrics = {
          psnr: psnr !== null ? psnr : '--',
          ssim: ssim !== null ? ssim : '--',
          sharpness: sharpness !== null ? sharpness : '--',
          scale_factor: '8x',
        };
      }

      const windowPreset = btn.dataset.window || null;
      const sliceIndex = btn.dataset.slice !== '' && btn.dataset.slice !== undefined ? btn.dataset.slice : null;
      const reportUrl = btn.dataset.reportUrl || null;
      const scaleFactor = parseInt(btn.dataset.scale, 10) || 8;
      const dicomUrl = btn.dataset.dicomUrl || null;
      const deidentified = btn.dataset.deidentified === 'true';
      const modality = btn.dataset.modality || 'general';

      modal.classList.remove('hidden');
      renderComparisonViewer(modalContainer, original, result, 'Interactive comparison inspector', title, metrics, windowPreset, sliceIndex, reportUrl, scaleFactor, dicomUrl, deidentified, modality);
    });
  });
});

