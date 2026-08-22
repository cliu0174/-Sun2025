# PI-MSCL 原生形状版：Conv1D 前表面色块紧凑修订

本目录根据作者的纠正说明制作，未覆盖此前版本。此次只调整 Conv1D 九组特征图最前表面中已有的小色块，不给后方衬底层新增任何网格。

## 推荐文件

- `PI-MSCL_architecture_native_shapes_fully_ungrouped_conv_cells_tighter_20260818.pptx`
  - 完全取消分组，555 个顶层原生对象。
  - 每个方格、叠板、箭头、曲线点和文本框均可直接单击编辑。

- `PI-MSCL_architecture_native_shapes_conv_cells_tighter_20260818.pptx`
  - 视觉相同的语义分组版，适合按模块整体移动。

## 本轮调整

- 仅处理带 `data-gradient-rank` 的 126 个 Conv1D 前表面色块。
- 白色分隔描边宽度缩小至原始宽度的 22%。
- 为避免 PowerPoint 将极细线仍显示成完整屏幕像素，白色分隔线透明度同时限制为 48%，使可见留白进一步弱化。
- 后方四层衬底继续保持纯色叠板，不新增网格。
- Input 与 Fuse 后融合张量保持上一版“紧凑单元版”的设置；面板、箭头、文字、公式、曲线和布局不变。

## QA

- 原生形状 538、原生连接线 17、可编辑文本框 87。
- 完全取消分组版为 555 个顶层对象；图片对象与嵌入媒体均为 0。
- 分组版和完全取消分组版的 PowerPoint 实际导出逐像素一致。
- 相对 SVG 基准的 RGB MAE 为 3.7195，全部硬检查通过。
- 独立视觉复核确认：仅 Conv1D 前表面白缝变细，各格仍可辨；后层无新增网格；Input、Fuse张量及其他模块未误改。

完整审计见 `qa/native_pptx_final_audit.json`。

## 口径提醒

本图仍保留参考设计稿中的 `40 × 14` 和 `L_rate`，只调整视觉和 PowerPoint 编辑结构；不改变正式研究的 16 维输入与归档实验 `smoothness_weight=0` 口径，且尚未插入主稿。
