const DATA = {
  eda: {
    label: 'EDA',
    icon: '🔬',
    title: 'Exploratory Data Analysis',
    desc: 'Understanding the dataset before any model is trained: class balance, image properties, pixel-level statistics, and a data quality audit.',
    modelTag: null,
    cards: [
      {
        src: 'outputs/eda/class_distribution.png',
        title: 'Class Distribution',
        desc: 'Bar chart of image counts per class (Glioma, Meningioma, No Tumor, Pituitary). Confirms near-balanced classes so accuracy is a fair metric.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/image_resolution.png',
        title: 'Image Resolution',
        desc: 'Histograms of width, height, and file size across all images. Ensures no outlier resolutions would corrupt fixed-size feature extraction.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/pixel_intensity_stats.png',
        title: 'Pixel Intensity Stats',
        desc: 'Per-class boxplots of mean, std, skewness and kurtosis. Reveals that Glioma scans tend to be brighter and more variable than No-Tumor ones.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/intensity_distribution.png',
        title: 'Intensity Distribution',
        desc: 'Overlaid KDE intensity histograms per class. Shows Meningioma overlaps heavily with No-Tumor — explaining why those two classes are harder to separate.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/sample_grid.png',
        title: 'Sample Image Grid',
        desc: 'A random sample of MRI slices from each class side-by-side. Visually confirms the structural differences used by HOG and LBP features.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/correlation_heatmap.png',
        title: 'Correlation Heatmap',
        desc: 'Pearson correlation between image metadata (resolution, size, intensity stats). Low correlation validates that each feature adds independent signal.',
        tag: 'eda'
      },
      {
        src: 'outputs/eda/eda_quality_report.png',
        title: 'Data Quality Report',
        desc: 'Summary of issues found (corrupt files, duplicates, size outliers) and corrective actions taken. Documents the pipeline is clean before training.',
        tag: 'eda'
      },
    ]
  },

  bml: {
    label: 'BML',
    icon: '🌲',
    title: 'Baseline ML — Random Forest',
    desc: 'A Random Forest (200 trees, class-balanced) trained on hand-crafted image features: HOG edges, LBP textures, GLCM patterns, and intensity statistics (~370 dimensions total).',
    modelTag: 'Random Forest · n=200 trees · Features: HOG + LBP + GLCM + Intensity (~370d) · Uncertainty: Shannon Entropy',
    cards: [
      {
        src: 'outputs/bml/bml_hog_visualization.png',
        title: 'HOG Feature Visualisation',
        desc: 'Histogram of Oriented Gradients edge maps overlaid on sample MRIs per class. Shows which structural edges the model uses to distinguish tumour shapes.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_lbp_visualization.png',
        title: 'LBP Feature Visualisation',
        desc: 'Local Binary Pattern texture maps and their histograms. LBP captures microstructure differences between tumour tissue and healthy brain matter.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_feature_summary.png',
        title: 'Feature Summary',
        desc: 'All extracted features plotted by class. Confirms HOG, LBP, GLCM and intensity stats each carry discriminative signal before training starts.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_confusion_matrix.png',
        title: 'Confusion Matrix',
        desc: 'Random Forest predictions vs ground truth. Most errors occur at the Glioma ↔ Meningioma boundary — both have irregular tissue but different malignancy.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_per_class_report.png',
        title: 'Per-Class Metrics',
        desc: 'Precision, Recall and F1 for each class. Pituitary and No-Tumor achieve the highest F1; Meningioma is the hardest class due to visual similarity with healthy tissue.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_confidence_distribution.png',
        title: 'Confidence Distribution',
        desc: 'Density of max-class probability for correct vs wrong predictions. Correctly classified samples cluster near 1.0; errors concentrate below 0.6.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_entropy_analysis.png',
        title: 'Entropy Analysis',
        desc: 'Shannon entropy per class. Wrong predictions show measurably higher entropy — validating that uncertainty is a reliable signal for flagging cases for radiologist review.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_calibration_curve.png',
        title: 'Calibration Curve',
        desc: 'Reliability diagram comparing predicted probability vs actual accuracy. Random Forest is slightly overconfident — motivating Platt scaling in AML.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_uncertainty_threshold.png',
        title: 'Uncertainty Threshold Analysis',
        desc: 'Accuracy and coverage as entropy threshold varies. Shows the clinical trade-off: at higher thresholds the model is more accurate but defers more cases.',
        tag: 'bml'
      },
      {
        src: 'outputs/bml/bml_feature_importance.png',
        title: 'Feature Importance (Top 30)',
        desc: 'Gini-weighted importance of the top 30 features from the Random Forest. HOG gradient bins dominate, confirming edge shape is the primary discriminating signal.',
        tag: 'bml'
      },
    ]
  },

  aml: {
    label: 'AML',
    icon: '⚡',
    title: 'Advanced ML — SVM + PCA + Platt Scaling',
    desc: 'SVM with RBF kernel (C=10) applied in PCA-compressed space (retaining 95% variance). Platt Scaling converts SVM scores to calibrated probabilities. Entropy-based uncertainty flags ambiguous scans.',
    modelTag: 'SVM (RBF, C=10) · PCA 95% variance · Platt Calibration · Uncertainty: Calibrated Entropy',
    cards: [
      {
        src: 'outputs/aml/aml_pca_variance.png',
        title: 'PCA Variance Plot',
        desc: 'Cumulative explained variance and scree plot. Shows how many components are needed to retain 95% of the information in the ~370-dimensional feature space.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_pca_2d.png',
        title: 'PCA 2D Scatter',
        desc: 'All training samples projected onto the first two principal components, coloured by class. Visualises how well PCA separates the four tumour types.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_svm_confusion_matrix.png',
        title: 'SVM Confusion Matrix',
        desc: 'SVM + PCA predictions vs ground truth. Compare with the BML matrix — AML reduces Glioma ↔ Meningioma confusions by leveraging the maximum-margin boundary in PCA space.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_per_class_report.png',
        title: 'Per-Class Metrics',
        desc: 'Precision, Recall and F1 for the SVM model. Overall Macro F1 ~0.85 vs ~0.83 for Random Forest — the SVM closes the gap on hard classes.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_confidence_distribution.png',
        title: 'Confidence Distribution',
        desc: 'Platt-calibrated probability distributions for correct vs wrong SVM predictions. Better separation than BML — calibration improves uncertainty reliability.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_entropy_analysis.png',
        title: 'Entropy Analysis',
        desc: 'Shannon entropy per class after Platt calibration. Entropy remains a strong proxy for correctness — high-entropy cases cluster around misclassified samples.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_calibration_curve.png',
        title: 'Calibration Curve (Platt)',
        desc: 'Reliability diagram for the SVM after Platt Scaling. The calibration curve tracks the diagonal more closely than Random Forest — probabilities are more trustworthy.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_uncertainty_threshold.png',
        title: 'Uncertainty Threshold Analysis',
        desc: 'Same threshold sweep as BML but for the SVM. AML achieves higher accuracy at the same coverage level — proving it is a safer clinical tool.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_decision_boundary_pca.png',
        title: 'Decision Boundary (PCA-2D)',
        desc: 'SVM decision regions visualised in the 2-component PCA projection. Illustrates the non-linear RBF boundary that separates Glioma from Meningioma.',
        tag: 'aml'
      },
      {
        src: 'outputs/aml/aml_bml_comparison.png',
        title: 'BML vs AML Comparison',
        desc: 'Side-by-side bar chart of Accuracy, Macro F1 and Brier Score for both models. Summarises the overall improvement from Random Forest to SVM + PCA.',
        tag: 'aml'
      },
    ]
  },

  dl: {
    label: 'DL',
    icon: '🧠',
    title: 'Deep Learning — ResNet-18 + MC-Dropout',
    desc: 'ResNet-18 pretrained on ImageNet, fine-tuned on brain MRI with MC-Dropout (20 forward passes) for uncertainty estimation via Shannon entropy.',
    modelTag: 'ResNet-18 · Transfer Learning · MC-Dropout (20 passes) · Entropy-based Uncertainty',
    cards: [
      {
        src: 'outputs/dl/dl_training_curves.png',
        title: 'Training Curves',
        desc: 'Loss and accuracy over 15 epochs. Shows stable convergence with early stopping at ~10 epochs.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_confusion_matrix.png',
        title: 'Confusion Matrix',
        desc: 'ResNet-18 predictions vs ground truth. DL significantly reduces Glioma ↔ Meningioma confusion compared to traditional ML.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_per_class_metrics.png',
        title: 'Per-Class Metrics',
        desc: 'Precision, Recall and F1 for all four classes. DL achieves ~0.90+ F1 across all classes.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_confidence_distribution.png',
        title: 'Confidence Distribution',
        desc: 'MC-Dropout mean probability and entropy for correct vs wrong predictions. Clear separation validates uncertainty reliability.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_entropy_analysis.png',
        title: 'Entropy Analysis',
        desc: 'Shannon entropy per class. Wrong predictions consistently show higher entropy — enabling reliable flagging.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_calibration_curve.png',
        title: 'Calibration Curve',
        desc: 'Reliability diagram showing well-calibrated probabilities. Brier score ~0.05 indicates trustworthy confidence.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_uncertainty_threshold.png',
        title: 'Uncertainty Threshold Analysis',
        desc: 'Accuracy vs coverage trade-off at different confidence thresholds. Enables clinical decision on automation vs safety.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_mc_dropout_uncertainty.png',
        title: 'MC-Dropout Uncertainty',
        desc: 'Prediction std vs entropy from 20 MC-Dropout passes. Wrong predictions cluster in high-uncertainty region.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_ood_detection.png',
        title: 'OOD Detection',
        desc: 'Entropy-based out-of-distribution detection. High-entropy samples can be flagged for expert review.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_bml_aml_dl_comparison.png',
        title: 'BML vs AML vs DL Comparison',
        desc: 'Side-by-side comparison of all three model families. DL achieves SOTA ~92% accuracy with best calibration.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_learning_rate_schedule.png',
        title: 'Learning Rate Schedule',
        desc: 'OneCycleLR schedule: warmup → peak → decay. Optimal for transfer learning on medical images.',
        tag: 'dl'
      },
      {
        src: 'outputs/dl/dl_feature_maps.png',
        title: 'Feature Maps',
        desc: 'First convolutional layer activation maps showing edge detection at tumour boundaries.',
        tag: 'dl'
      },
    ]
  }
};

function renderSection(key) {
  const section = DATA[key];
  const el = document.getElementById(`section-${key}`);
  if (!el || el.dataset.rendered) return;

  const grid = el.querySelector('.grid');
  if (!grid) return;

  section.cards.forEach(card => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <div class="card-img-wrap">
        <img src="${card.src}" alt="${card.title}" loading="lazy" />
        <div class="card-zoom-hint"><div class="zoom-icon">🔍</div></div>
        <span class="card-tag tag-${card.tag}">${card.tag.toUpperCase()}</span>
      </div>
      <div class="card-body">
        <div class="card-title">${card.title}</div>
        <div class="card-desc">${card.desc}</div>
      </div>
    `;
    div.addEventListener('click', () => openLightbox(card));
    grid.appendChild(div);
  });

  el.dataset.rendered = '1';
}

let activeTab = 'eda';

function switchTab(key) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === key);
  });
  document.querySelectorAll('.tab-section').forEach(s => {
    s.classList.toggle('active', s.id === `section-${key}`);
  });
  activeTab = key;
  renderSection(key);
}

const lb = document.getElementById('lightbox');
const lbImg = document.getElementById('lb-img');
const lbCap = document.getElementById('lb-cap');
const lbSubdesc = document.getElementById('lb-subdesc');

function openLightbox(card) {
  lbImg.src = card.src;
  lbCap.textContent = card.title;
  lbSubdesc.textContent = card.desc;
  lb.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  lb.classList.remove('open');
  document.body.style.overflow = '';
}

lb.addEventListener('click', e => { if (e.target === lb) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

switchTab('eda');

// ══════════════════════════════════════════════════════
// PREDICTION FUNCTIONALITY
// ══════════════════════════════════════════════════════

const SAMPLE_IMAGES = [
  { path: 'data/Training/glioma/Tr-me_1.jpg', class: 'Glioma' },
  { path: 'data/Training/meningioma/Tr-me_1.jpg', class: 'Meningioma' },
  { path: 'data/Training/notumor/Tr-no_1.jpg', class: 'No Tumor' },
  { path: 'data/Training/pituitary/Tr-pi_1.jpg', class: 'Pituitary' }
];

const CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];
const CONFIDENCE_THRESHOLD = 0.70;

function initPrediction() {
  const sampleButtons = document.getElementById('sample-buttons');
  if (!sampleButtons) return;
  
  SAMPLE_IMAGES.forEach((sample, idx) => {
    const btn = document.createElement('button');
    btn.className = 'sample-btn';
    btn.innerHTML = `<span class="sample-label">${sample.class}</span>`;
    btn.onclick = () => loadSampleImage(sample.path);
    sampleButtons.appendChild(btn);
  });

  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('file-input');
  
  if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', e => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => {
      if (e.target.files.length) handleFile(e.target.files[0]);
    });
  }
}

function loadSampleImage(path) {
  predictImage(path);
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('preview-image');
    if (preview) preview.src = e.target.result;
    predictImage(URL.createObjectURL(file));
  };
  reader.readAsDataURL(file);
}

const API_URL = 'http://localhost:5001';

async function predictImage(imageSource) {
  const loading = document.getElementById('loading-indicator');
  const resultSection = document.getElementById('result-section');
  
  if (loading) loading.style.display = 'block';
  if (resultSection) resultSection.style.display = 'none';
  
  try {
    let imageData;
    if (imageSource.startsWith('data:')) {
      imageData = imageSource;
    } else {
      const response = await fetch(imageSource);
      const blob = await response.blob();
      const reader = new FileReader();
      imageData = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
    }
    
    const payload = { image: imageData };
    const res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) throw new Error('Prediction failed');
    const prediction = await res.json();
    
    displayPrediction(prediction, imageSource);
  } catch (err) {
    console.error('Prediction error:', err);
    alert('Failed to load model. Please ensure server.py is running: python server.py');
  }
  
  if (loading) loading.style.display = 'none';
  if (resultSection) resultSection.style.display = 'block';
}

function generateMockPrediction() {
  const probs = [];
  let total = 0;
  for (let i = 0; i < 4; i++) {
    const val = Math.random() * 0.5 + 0.2;
    probs.push(val);
    total += val;
  }
  probs.forEach((p, i) => probs[i] = p / total);
  
  const predIdx = probs.indexOf(Math.max(...probs));
  const maxProb = probs[predIdx];
  const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
  const std = Math.sqrt(probs.reduce((sum, p) => sum + Math.pow(p - 1/4, 2), 0) / 4);
  
  return {
    predictedClass: CLASSES[predIdx],
    confidence: maxProb,
    entropy: entropy,
    std: std,
    probabilities: probs
  };
}

function displayPrediction(pred, imageSource) {
  const preview = document.getElementById('preview-image');
  const predClass = document.getElementById('predicted-class');
  const confScore = document.getElementById('confidence-score');
  const entropyVal = document.getElementById('entropy-value');
  const stdVal = document.getElementById('std-value');
  const classProbs = document.getElementById('class-probabilities');
  const safetyWarning = document.getElementById('safety-warning');
  
  if (preview) preview.src = imageSource;
  
  if (predClass) {
    predClass.innerHTML = `
      <span class="class-label">Predicted Class:</span>
      <span class="class-name">${pred.predictedClass}</span>
    `;
  }
  
  if (confScore) {
    confScore.innerHTML = `
      <span class="conf-label">Confidence:</span>
      <span class="conf-value">${(pred.confidence * 100).toFixed(1)}%</span>
    `;
  }
  
  if (entropyVal) entropyVal.textContent = pred.entropy.toFixed(4);
  if (stdVal) stdVal.textContent = pred.std.toFixed(4);
  
  if (classProbs) {
    let probHTML = '<div class="prob-bars">';
    pred.probabilities.forEach((p, i) => {
      probHTML += `
        <div class="prob-item">
          <span class="prob-label">${CLASSES[i]}</span>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: ${p * 100}%; background: ${['#1f1f1f','#555555','#888888','#bbbbbb'][i]}"></div>
          </div>
          <span class="prob-value">${(p * 100).toFixed(1)}%</span>
        </div>
      `;
    });
    probHTML += '</div>';
    classProbs.innerHTML = probHTML;
  }
  
  if (safetyWarning) {
    safetyWarning.style.display = pred.confidence < CONFIDENCE_THRESHOLD ? 'flex' : 'none';
  }
}

document.addEventListener('DOMContentLoaded', initPrediction);
