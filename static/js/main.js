const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const selectImageButton = document.getElementById('select-image-button');
const resultSection = document.getElementById('result-section');
const ACCEPTED_FORMATS = ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'];

let selectedFile = null;

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

  try {
    const response = await fetch('/upload/', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
      },
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      resultSection.classList.remove('hidden');
      resultSection.innerHTML = `
        <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
          <p class="font-semibold">Upload failed.</p>
          <p class="mt-2 text-sm text-slate-400">${data.error || 'Please try again with a valid medical image file.'}</p>
        </div>
      `;
      return;
    }

    resultSection.classList.remove('hidden');
    resultSection.innerHTML = `
      <div class="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-2xl">
        <h2 class="text-3xl font-semibold mb-6 text-center">✅ QGAN Enhanced Result</h2>
        <div class="grid gap-4 lg:grid-cols-2">
          <div class="rounded-3xl overflow-hidden border border-slate-800 bg-slate-950">
            <div class="px-4 py-3 bg-slate-900 text-sm text-slate-400">Original image</div>
            <img src="${data.original}" alt="Original image" class="w-full h-80 object-contain bg-black">
          </div>
          <div class="rounded-3xl overflow-hidden border border-slate-800 bg-slate-950">
            <div class="px-4 py-3 bg-slate-900 text-sm text-emerald-300">Enhanced image</div>
            <img src="${data.result}" alt="Enhanced image" class="w-full h-80 object-contain bg-black">
          </div>
        </div>
        <p class="mt-6 text-center text-emerald-300 font-medium">${data.message}</p>
      </div>
    `;
  } catch (error) {
    resultSection.classList.remove('hidden');
    resultSection.innerHTML = `
      <div class="rounded-3xl bg-rose-950/90 border border-rose-800 p-6 text-center text-rose-200">
        <p class="font-semibold">Upload failed.</p>
        <p class="mt-2 text-sm text-slate-400">Please try again with a valid medical image file.</p>
      </div>
    `;
  }
}
