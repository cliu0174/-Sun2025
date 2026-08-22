# PI-MSCL 原生形状版：后层边框淡化修订

本目录是原生 PowerPoint 形状版本的后续修订，未覆盖上一版。修订重点是立体叠板的边框透明度：除最前表面和最终外轮廓外，后方衬底层的填充与边框现在使用同一层级透明度，由后向前逐层增强。

## 推荐文件

- `PI-MSCL_architecture_native_shapes_fully_ungrouped_light_backplanes_20260818.pptx`
  - 完全取消分组，555 个顶层原生对象。
  - 每个方格、叠板、箭头、曲线点和文本框均可直接单击编辑。

- `PI-MSCL_architecture_native_shapes_light_backplanes_20260818.pptx`
  - 视觉相同的语义分组版，适合按模块整体移动。

## 可编辑性与视觉检查

- 原生形状：538
- 原生连接线：17
- 可编辑文本框：87
- 图片对象与嵌入媒体：0
- 分组版与完全取消分组版的 PowerPoint 实际导出逐像素一致。
- 相对 SVG 基准的 RGB MAE 从上一版的 3.9507 降至 3.7638。
- 独立视觉复核确认，Conv1D 九组叠板和 Fuse 后融合张量均实现后层边框逐层变淡，且未出现边框消失、层次塌陷或前后关系反转。

完整审计见 `qa/native_pptx_final_audit.json`。

## 口径提醒

本图继续保留参考设计稿中的 `40 × 14` 和 `L_rate` 表达，只更新视觉与 PowerPoint 编辑结构；它不改变正式研究的 16 维输入和归档实验 `smoothness_weight=0` 的实现口径，且尚未插入主稿。
