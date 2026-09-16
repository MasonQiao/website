const slider = document.getElementById("z-slider");
const layerImage = document.getElementById("layer-image");
const secondaryLayerImage = document.getElementById("secondary-layer-image");
const secondaryFigure = document.getElementById("secondary-figure");
const primaryCaption = document.getElementById("primary-caption");
const secondaryCaption = document.getElementById("secondary-caption");
const imageGrid = document.getElementById("image-grid");
const viewerRoot = document.getElementById("viewer-root");
const label = document.getElementById("slider-label");
const minimum = document.getElementById("minimum");
const maximum = document.getElementById("maximum");
const currentValue = document.getElementById("current-value");

let imageSources = [];
let secondaryImageSources = [];
let storageKey = null;
let lastFrameHeight = null;
let frameRequest = null;

function sendMessage(type, data = {}) {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type,
      ...data,
    },
    "*",
  );
}

function setFrameHeight() {
  window.cancelAnimationFrame(frameRequest);
  frameRequest = window.requestAnimationFrame(() => {
    const height = Math.ceil(viewerRoot.getBoundingClientRect().height);
    if (height <= 0 || height === lastFrameHeight) {
      return;
    }
    lastFrameHeight = height;
    sendMessage("streamlit:setFrameHeight", { height });
  });
}

function saveSelectedValue() {
  if (storageKey) {
    try {
      window.sessionStorage.setItem(storageKey, slider.value);
    } catch {
      // The viewer still works when browser privacy settings disable storage.
    }
  }
}

function storedValue(args) {
  let stored = Number.NaN;
  if (storageKey) {
    try {
      stored = Number.parseInt(window.sessionStorage.getItem(storageKey), 10);
    } catch {
      stored = Number.NaN;
    }
  }
  if (
    Number.isInteger(stored)
    && stored >= args.min_value
    && stored <= args.max_value
  ) {
    return stored;
  }
  return args.value;
}

function updateDisplayedValue() {
  currentValue.value = `${slider.value}/${slider.max}`;
  const sourceIndex = Number.parseInt(slider.value, 10) - 1;
  if (imageSources[sourceIndex] && layerImage.src !== imageSources[sourceIndex]) {
    layerImage.src = imageSources[sourceIndex];
  }
  if (
    secondaryImageSources[sourceIndex]
    && secondaryLayerImage.src !== secondaryImageSources[sourceIndex]
  ) {
    secondaryLayerImage.src = secondaryImageSources[sourceIndex];
  }
  layerImage.alt = `${primaryCaption.textContent} z-layer ${slider.value} of ${slider.max}`;
  secondaryLayerImage.alt = `${secondaryCaption.textContent} z-layer ${slider.value} of ${slider.max}`;
  slider.setAttribute(
    "aria-valuetext",
    `z-layer ${slider.value} of ${slider.max}`,
  );
}

slider.addEventListener("input", () => {
  updateDisplayedValue();
  saveSelectedValue();
});

slider.addEventListener("change", () => {
  updateDisplayedValue();
  saveSelectedValue();
});

function applyTheme(theme) {
  if (!theme) {
    return;
  }
  document.body.style.color = theme.textColor;
  document.body.style.fontFamily = theme.font;
  if (theme.primaryColor) {
    document.documentElement.style.setProperty(
      "--primary-color",
      theme.primaryColor,
    );
  }
  if (theme.base) {
    document.documentElement.style.colorScheme = theme.base;
  }
}

function render(args, theme) {
  applyTheme(theme);
  label.textContent = "Z-layer";
  primaryCaption.textContent = args.label;
  secondaryCaption.textContent = args.secondary_label || "";
  imageSources = args.sources;
  secondaryImageSources = args.secondary_sources || [];
  secondaryFigure.hidden = secondaryImageSources.length === 0;
  imageGrid.style.gridTemplateColumns = secondaryImageSources.length
    ? "repeat(2, minmax(0, 1fr))"
    : "minmax(0, 1fr)";
  storageKey = args.storage_key;
  slider.min = args.min_value;
  slider.max = args.max_value;
  minimum.textContent = args.min_value;
  maximum.textContent = args.max_value;
  slider.setAttribute("aria-label", args.label);
  slider.value = storedValue(args);
  layerImage.style.aspectRatio = `${args.image_width} / ${args.image_height}`;
  secondaryLayerImage.style.aspectRatio = `${args.image_width} / ${args.image_height}`;
  updateDisplayedValue();
  setFrameHeight();
}

window.addEventListener("message", (event) => {
  if (event.data?.type === "streamlit:render") {
    render(event.data.args, event.data.theme);
  }
});

layerImage.addEventListener("load", setFrameHeight);
secondaryLayerImage.addEventListener("load", setFrameHeight);
window.addEventListener("resize", setFrameHeight);
new ResizeObserver(setFrameHeight).observe(viewerRoot);

sendMessage("streamlit:componentReady", { apiVersion: 1 });
