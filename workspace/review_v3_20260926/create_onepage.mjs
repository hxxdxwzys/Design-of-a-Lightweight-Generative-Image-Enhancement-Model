import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const runtime = 'C:/Users/Yuki Nagato/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES = path.join(runtime, 'node/node_modules');
const skill = 'C:/Users/Yuki Nagato/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(runtime, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const workspaceDir = 'D:/models/review_V3_20260926/ppt_work';
const tmp = path.join(workspaceDir, 'private');
const output = path.join(workspaceDir, 'output');
await fs.mkdir(tmp, { recursive: true });
await fs.mkdir(output, { recursive: true });
const family = resolvePresentationFont({ fontFamily: 'Microsoft YaHei', availableFonts: ['Microsoft YaHei'] });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = presentation.slides.add();
slide.background.fill = '#FFFFFF';
function text(name, value, left, top, width, height, size=24, bold=false, color='#233547') {
  const s = slide.shapes.add({ geometry:'textbox', name, position:{left,top,width,height}, fill:'none', line:{fill:'none',width:0} });
  s.text = value;
  s.text.style = { typeface:family, fontSize:size, bold, color, autoFit:'none' };
  return s;
}
text('title','TinyRef V3 轻量一步扩散图像增强',48,28,1190,76,46,true,'#102A43');
text('subtitle','赛道二初赛复核　4K同分辨率增强　对应初赛V3原件',50,111,1180,45,24,false,'#506579');

text('method-heading','一步扩散与细化',50,183,565,50,32,true,'#0A6473');
text('method','512图块下采样至256，UNet在t=500预测噪声。显式一步去噪后上采样回512，再经细化CNN与内置锐化输出。',50,241,565,125,24);
text('params','876,662 个参数',50,367,565,49,32,true,'#102A43');
text('params-detail','UNet 871,139　细化CNN 5,523\n原模型权重fp16，整链TorchScript。\n原runner在CUDA支持时使用CUDA Graph。',50,425,565,115,24);
text('tiling','4K采用512窗口、32重叠与羽化融合。\n4096×3072共63块，各块与512接口\n调用同一模型，输出保持原分辨率。',50,548,585,110,23);

text('verification-heading','原件与复现实证',710,183,520,50,32,true,'#0A6473');
text('identity','模型与runner保留V3原件字节。\ncase3、case17的参考图直接取自\n初赛原ZIP，未替换为后续V4t1结果。',710,241,520,116,24);
text('result','双进程复现，JPEG字节一致',710,367,520,50,30,true,'#102A43');
text('result-detail','两独立进程分别重生成两个完整样例，\n四次均与原JPEG逐字节一致。\n双方解码RGB最大差和平均差均为0。',710,426,520,115,24);
text('scope','实测仅覆盖这两张图及记录的CUDA环境。\n保留原边界处理，不改写历史作品。\n原件指纹、环境与复现证据随工程提供。',710,548,520,110,22);
text('footer','队伍 hxxdxwzys　联系方式 18224070607　2026-09-26　　平台成绩与参赛资格以官方记录及复核为准',50,676,1180,30,17,false,'#506579');

slide.speakerNotes.textFrame.setText([
  '版本：Track2_TinyRefV3.zip。此页不使用V4t1指标，不声称赛后版本继承官方成绩。',
  '源ZIP SHA256: d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712。',
  '模型SHA256: 421be231d504b1e1d20656ff7aa1c45a377434ff4762cc79c05416a2c7b5306d。',
  '参数来源: verification/model_audit.json。原件指纹: manifest.json / original_members.json。',
  '结构来源: 原TorchScript与model_dir/runner.py，历史opt11_render_pkg.py。',
  '复现来源: verification/process_1_results.json、process_2_results.json、independent_comparison.json。',
  '运行环境: Python3.13.0、torch2.7.0+cu126、CUDA12.6、RTX4060 Laptop、Pillow12.3.0。',
  '保留外缘零羽化权重带来的历史暗边。该复现不证明其余98图已重新推理或官方环境兼容性。',
  '联系方式沿用用户既有Proposal_OnePage.pptx中的信息。',
].join('\n'));
const candidatePath = path.join(tmp,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath = path.join(output,'Proposal_OnePage_V3.pptx');
const result = await finalizePresentation({
  explicitTotalSlideCount:1, requiredNativeTableOwnerSlides:[], requiredNativeChartOwnerSlides:[],
  workspaceDir, candidatePath, finalPath,
  pythonExecutable:path.join(runtime,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,
  receiptPath:path.join(tmp,'validation.json'),
});
console.log(JSON.stringify(result));
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const finalSlide = finalDeck.slides.items[0];
const preview = await finalDeck.export({slide:finalSlide,format:'png',scale:1.5});
await fs.writeFile(path.join(output,'Proposal_OnePage_V3.png'),new Uint8Array(await preview.arrayBuffer()));
const layout = await finalSlide.export({format:'layout'});
await fs.writeFile(path.join(tmp,'final.layout.json'),await layout.text());
console.log(finalPath);
