import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/Yuki Nagato/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const skill='C:/Users/Yuki Nagato/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const workspaceDir='D:/models/quality27_20260927/ppt_work';
const tmp=path.join(workspaceDir,'private'), output=path.join(workspaceDir,'output');
await fs.mkdir(tmp,{recursive:true});await fs.mkdir(output,{recursive:true});
const family=resolvePresentationFont({fontFamily:'Microsoft YaHei',availableFonts:['Microsoft YaHei']});
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
function text(name,value,left,top,width,height,size=24,bold=false,color='#233547'){
  const s=slide.shapes.add({geometry:'textbox',name,position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  s.text=value;s.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none'};
}
text('title','TinyRef V4t1 轻量一步扩散增强',48,28,1190,76,46,true,'#102A43');
text('subtitle','赛后工程修订版 EdgeFix　2026-09-27　官方成绩及换件许可未确认',50,111,1180,45,24,false,'#82551A');
text('method-heading','同一模型完成测速与出图',50,183,565,50,32,true,'#0A6473');
text('method','512图块下采样至256，UNet在t=500预测噪声。一步去噪后上采样回512，细化CNN与内置锐化生成输出。',50,241,565,125,24);
text('params','876,662 个参数',50,367,565,49,32,true,'#102A43');
text('params-detail','保留V4t1模型与runner原字节。\n权重fp16，整链TorchScript。\nCUDA Graph覆盖完整512推理路径。',50,425,565,115,24);
text('tiling','4K采用512窗口与32重叠。\n4096×3072共63块，逐块调用同一模型。\n输入输出保持原分辨率，无图片检索。',50,548,585,110,23);
text('verification-heading','仅修正出图边缘与整数恢复',710,183,520,50,32,true,'#0A6473');
text('identity','外缘权重改为1，内部重叠继续羽化。\n融合结果按最近整数恢复8位色阶。\n消除外缘零权重导致的黑线。',710,241,520,116,24);
text('result','两样例跨进程复现一致',710,367,520,50,30,true,'#102A43');
text('result-detail','case3、case17完整4K重新生成，\n独立进程复现JPEG字节一致。\n逐图清单绑定模型、runner与推理入口。',710,426,520,115,24);
text('scope','本版不继承V3的官方分数。\n本轮条件细化器训练未达质量门槛，\n新权重未纳入工程。V3原件另行保留。',710,548,520,110,22);
text('footer','队伍 hxxdxwzys　联系方式 18224070607　　仅作工程修订说明，是否接收赛后修改由组委会决定',50,676,1180,30,17,false,'#506579');
slide.speakerNotes.textFrame.setText([
  '本页对应V4t1_EdgeFix_20260927，不是初赛V3原得分版本。',
  '模型SHA256:2770215987e2b5585dc26870a8a6fee05672bbf514267336498a1bf9e6409b62。',
  'runner SHA256:8e434e0240eed20b924e04f5f7338a037c8ec7821b8067159fe50ac7492ff70b。',
  '复现证据:verification/edge_phase1.json、edge_phase2.json、generation_manifest.json。',
  '两样例及本地CUDA环境范围内验证。不代表100图重生成或官方环境通过。',
  '全图量化前/编码前除边缘16像素带外，与同一批模型块旧融合结果最大差不超过1色阶。JPEG重编码可以传播局部差异。',
  '未声称整体画质、泛化能力或官方分数提高。初赛后换件许可UNKNOWN。',
  '原始队伍与联系方式沿用既有方案简介。'
].join('\n'));
const candidatePath=path.join(tmp,'candidate.pptx');
await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(output,'Proposal_OnePage_V4t1_EdgeFix.pptx');
console.log(JSON.stringify(await finalizePresentation({explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  workspaceDir,candidatePath,finalPath,pythonExecutable:path.join(runtime,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')})));
const deck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await deck.export({slide:deck.slides.items[0],format:'png',scale:1.5});
await fs.writeFile(path.join(output,'Proposal_OnePage_V4t1_EdgeFix.png'),new Uint8Array(await png.arrayBuffer()));
console.log(finalPath);
