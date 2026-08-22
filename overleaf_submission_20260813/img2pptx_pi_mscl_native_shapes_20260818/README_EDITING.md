# PI-MSCL 原生 PowerPoint 形状版本

本目录基于上一版 `img2pptx` 的 `full.svg` 继续转换，PPTX 内不再嵌入整张 SVG 或位图，而是使用原生 PowerPoint/DrawingML 对象。

## 建议使用的两个版本

- `PI-MSCL_architecture_native_shapes_20260818.pptx`
  - 推荐日常编辑。
  - 主要模块按 Input、三条 CNN 支路、Fuse、LSTM、回归头、输出曲线和三条损失路径分组。
  - PowerPoint 顶层有 29 个对象；递归展开后有 580 个命名对象。
  - 双击某个模块组即可进入组内编辑单个方格、文字、箭头或节点，也可对模块执行一次“取消组合”。

- `PI-MSCL_architecture_native_shapes_fully_ungrouped_20260818.pptx`
  - 适合需要直接单击任意小方格、曲线点或文字的场景。
  - 共有 555 个顶层原生对象，无分组、无图片对象。

## 可编辑性审计

- 原生形状：538
- 原生连接线：17
- 可编辑文本框：87
- 自定义矢量几何：278
- 图片对象：0
- 嵌入媒体文件：0

两个版本经 PowerPoint 实际打开并导出，渲染结果逐像素一致。完整审计位于 `qa/native_pptx_final_audit.json`，对象清单位于 `qa/native_editability_inventory.json`。

## 口径提醒

本图继续忠实保留参考设计稿中的 `40 × 14` 输入和 `L_rate` 三损失目标，只是改变 PowerPoint 的编辑方式；它不改变项目当前正式研究的 16 维输入与归档主实验中 `smoothness_weight=0` 的实现口径，也尚未插入主稿。
