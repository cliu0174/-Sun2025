const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const pptxgen = require('pptxgenjs');

const root = __dirname;
const fullSvg = path.join(root, 'full.svg');
const qaDir = path.join(root, 'qa');
const cropDir = path.join(qaDir, 'component_crops');
const modulesDir = path.join(root, 'modules');
fs.mkdirSync(qaDir, { recursive: true });
fs.mkdirSync(cropDir, { recursive: true });

async function renderSvg(svgPath, pngPath, width, height) {
  const opts = { density: 144 };
  let image = sharp(svgPath, opts).flatten({ background: '#FFFFFF' });
  if (width && height) image = image.resize(width, height, { fit: 'fill' });
  await image.png().toFile(pngPath);
}

async function main() {
  await renderSvg(fullSvg, path.join(qaDir, 'rendered_svg.png'), 1672, 941);
  const moduleFiles = fs.readdirSync(modulesDir).filter(f => f.endsWith('.svg'));
  for (const file of moduleFiles) {
    const inPath = path.join(modulesDir, file);
    const outPath = path.join(cropDir, file.replace(/\.svg$/i, '_render.png'));
    await renderSvg(inPath, outPath);
  }

  const pptx = new pptxgen();
  const slideW = 13.333333;
  const slideH = slideW * 941 / 1672;
  pptx.defineLayout({ name: 'SOURCE_RATIO', width: slideW, height: slideH });
  pptx.layout = 'SOURCE_RATIO';
  pptx.author = 'Codex / img2pptx';
  pptx.company = 'PI-MSCL project';
  pptx.subject = 'Editable reconstruction of PI-MSCL architecture';
  pptx.title = 'PI-MSCL architecture';
  pptx.lang = 'en-US';
  pptx.theme = {
    headFontFace: 'Arial',
    bodyFontFace: 'Arial',
    lang: 'en-US'
  };
  const slide = pptx.addSlide();
  slide.background = { color: 'FFFFFF' };
  slide.addImage({
    path: fullSvg,
    x: 0,
    y: 0,
    w: slideW,
    h: slideH,
    altText: 'PI-MSCL architecture and sparse-label physics-informed learning diagram'
  });
  await pptx.writeFile({ fileName: path.join(root, 'final.pptx') });
  console.log(`Rendered SVG and wrote final.pptx at ${slideW.toFixed(3)} x ${slideH.toFixed(3)} inches`);
}

main().catch(err => { console.error(err); process.exit(1); });
