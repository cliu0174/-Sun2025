/**
 * 生成第四章正文 docx
 * 格式：A4，宋体小四，1.5倍行距，首行缩进2字符
 * 表格数据留空，图位置用占位段落标注
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, TableLayoutType, LevelFormat
} = require('docx');
const fs = require('fs');

// ─── 尺寸常量 ─────────────────────────────────────────────────────────────────
const A4_W   = 11906; // DXA
const A4_H   = 16838;
const MARGIN = { top: 1800, bottom: 1800, left: 1800, right: 1260 }; // 3.2/2.2/3.2/2.2cm
const CONTENT_W = A4_W - MARGIN.left - MARGIN.right; // 8846

// ─── 辅助函数 ─────────────────────────────────────────────────────────────────
const FONT_CN = '宋体';
const FONT_EN = 'Times New Roman';
const SZ_BODY = 24;   // 小四 12pt (半点)
const SZ_H1   = 32;   // 三号 16pt
const SZ_H2   = 28;   // 四号 14pt
const SZ_H3   = 24;   // 小四 12pt bold

/** 正文段落（首行缩进2字符，两端对齐，1.5倍行距） */
function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({
    text,
    font: { name: FONT_CN, eastAsia: FONT_CN },
    size: SZ_BODY,
    ...opts.runOpts
  })];
  return new Paragraph({
    children: runs,
    spacing: { line: 360, lineRule: 'auto', before: 0, after: 0 },
    indent: opts.noIndent ? {} : { firstLineChars: 200, firstLine: 480 },
    alignment: opts.align || AlignmentType.BOTH,
    ...opts.paraOpts
  });
}

/** 多段 TextRun（用于混排中英文） */
function tr(text, bold = false, italic = false) {
  return new TextRun({
    text,
    font: { name: FONT_CN, eastAsia: FONT_CN },
    size: SZ_BODY,
    bold,
    italics: italic
  });
}
function tre(text, bold = false, italic = false) {
  return new TextRun({ text, font: { name: FONT_EN }, size: SZ_BODY, bold, italics: italic });
}

/** 居中公式段落（无缩进） */
function formula(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: { name: FONT_EN }, size: SZ_BODY, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, lineRule: 'auto', before: 120, after: 120 },
  });
}

/** 图/表占位段落（居中，灰色） */
function placeholder(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY, color: '888888' })],
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, lineRule: 'auto', before: 240, after: 240 },
  });
}

/** 图题 */
function figcap(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY })],
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, lineRule: 'auto', before: 60, after: 240 },
  });
}

/** 空行 */
function blank() {
  return new Paragraph({ children: [], spacing: { line: 360, lineRule: 'auto', before: 0, after: 0 } });
}

// ─── 表格工具 ─────────────────────────────────────────────────────────────────
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const BORDERS_ALL = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const BORDERS_NONE = {
  top: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
  bottom: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
  left: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
  right: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
};

function cell(text, opts = {}) {
  const { shade, bold, width, align, borders } = opts;
  return new TableCell({
    borders: borders || BORDERS_ALL,
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY, bold: !!bold })],
      alignment: align || AlignmentType.CENTER,
      spacing: { line: 300, lineRule: 'auto' },
    })]
  });
}

function headerRow(cols, widths, totalW) {
  return new TableRow({
    tableHeader: true,
    children: cols.map((c, i) => cell(c, { shade: 'D9E2F3', bold: true, width: widths ? widths[i] : Math.floor(totalW / cols.length) }))
  });
}

function dataRow(cols, widths, totalW, firstBold = false) {
  return new TableRow({
    children: cols.map((c, i) => cell(c, {
      bold: firstBold && i === 0,
      width: widths ? widths[i] : Math.floor(totalW / cols.length),
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER
    }))
  });
}

function makeTable(headers, rows, widths, totalW = CONTENT_W) {
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: widths || headers.map(() => Math.floor(totalW / headers.length)),
    rows: [
      headerRow(headers, widths, totalW),
      ...rows.map(r => dataRow(r, widths, totalW, true))
    ],
  });
}

function tablecap(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY })],
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, lineRule: 'auto', before: 240, after: 60 },
  });
}

// ─── 一、二、三级标题 ─────────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_H1, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, lineRule: 'auto', before: 480, after: 240 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_H2, bold: true })],
    alignment: AlignmentType.LEFT,
    spacing: { line: 360, lineRule: 'auto', before: 360, after: 120 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_H3, bold: true })],
    alignment: AlignmentType.LEFT,
    spacing: { line: 360, lineRule: 'auto', before: 240, after: 60 },
  });
}

/** 带编号的列表项（无首行缩进，左缩进） */
function listItem(num, text) {
  return new Paragraph({
    children: [
      new TextRun({ text: `${num}.　`, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY, bold: true }),
      new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY }),
    ],
    indent: { left: 480, hanging: 480 },
    alignment: AlignmentType.BOTH,
    spacing: { line: 360, lineRule: 'auto', before: 60, after: 60 },
  });
}

/** 条目说明段（接着列表项，带缩进） */
function listBody(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY })],
    indent: { left: 480 },
    alignment: AlignmentType.BOTH,
    spacing: { line: 360, lineRule: 'auto', before: 0, after: 60 },
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// 正文内容
// ═══════════════════════════════════════════════════════════════════════════════
const children = [];

// ────────────────────────────────────────────────────────────────────────────
// 章标题
children.push(h1('第四章  面向部分生命周期监督的多尺度物理一致性锂离子电池SOH估计方法'));

// ════════════════════════════════════════════════════════════════════════════
// 4.1 引言
// ════════════════════════════════════════════════════════════════════════════
children.push(h2('4.1  引言'));

children.push(p([tr('锂离子电池健康状态（'), tre('State of Health, SOH'), tr('）的准确评估是电池管理系统（'), tre('Battery Management System, BMS'), tr('）实现安全运行与寿命管理的关键基础。在长期服役过程中，电池内部会经历活性物质损失、电解液分解及'), tre('SEI'), tr('膜增厚等多种复杂的退化机制，其宏观表现为可用容量衰减与内阻增长，进而影响系统的可靠性与安全性。然而由于电池类型、使用工况及环境条件的高度多样性，'), tre('SOH'), tr('的稳定建模与精准估计仍面临诸多挑战。')]));

children.push(p([tr('近年来，学术界对'), tre('SOH'), tr('估计方法进行了广泛探索，主要可归纳为实验测量法、基于模型的方法以及数据驱动方法三类。实验测量法与等效电路模型虽在工程中应用广泛，但其精度高度依赖传感器的初始精度及复杂的参数辨识过程，在处理非线性衰退时存在局限性。随着深度学习的发展，基于长短期记忆网络（'), tre('LSTM'), tr('）、卷积神经网络（'), tre('CNN'), tr('）等算法的方法在'), tre('SOH'), tr('估计领域展现出强大的特征挖掘能力，能够有效建立电压、电流、温度等可观测信号与健康状态之间的非线性映射关系，避免了复杂电化学模型参数难以获取的问题。')]));

children.push(p([tr('然而，基础神经网络在实际工程应用中暴露出明显的共性瓶颈。这类模型本质上属于"黑盒"映射，其性能高度依赖训练数据的完整性与分布一致性。更重要的是，在真实'), tre('BMS'), tr('运行中，'), tre('SOH'), tr('标签通常并非连续可得——容量标定往往依赖离线测试、定期维护或抽检诊断，因此同一电池的大量服役循环可能只有可观测运行特征，而没有同步的'), tre('SOH'), tr('标签。若仍按照完整生命周期监督假设训练模型，容易高估算法在工程部署中的实际可用性。在缺乏物理约束的情况下，模型预测轨迹中还可能出现容量非单调回弹或高频随机波动等不符合物理规律的现象，削弱其在实际'), tre('BMS'), tr('部署中的可靠性。')]));

children.push(p([tr('物理一致性约束学习（'), tre('Physics-consistent learning'), tr('）为'), tre('SOH'), tr('估计提供了一种兼顾灵活性与稳定性的建模思路。该方法将电池退化过程中普遍成立的宏观物理先验——如容量衰减的单调性趋势与物理边界约束——通过定制化损失函数的形式隐式嵌入神经网络的训练过程，从而引导模型学习符合物理规律的退化演化模式。这种建模范式既保留了深度学习处理复杂非线性数据的能力，又通过物理规律锚定了模型的输出空间，引导模型学习更加合理、稳定的退化演化轨迹。')]));

children.push(p([tr('现有'), tre('SOH'), tr('估计研究大多从完整标签或密集标定条件出发，对"生命周期标签不完整"这一工程常态讨论不足。与此同时，传统单尺度卷积网络仅使用固定大小的卷积核，只能捕获单一粒度的充电动力学模式，在面对电池退化过程中同时存在的短程局部波动、中程阶段性变化与长程趋势性衰减时，特征表达能力受限。近年来，多尺度卷积结构在时序分析领域展现出显著优势，通过并行使用不同感受野的卷积核实现多粒度特征融合，但其在部分生命周期监督'), tre('SOH'), tr('估计中的作用机制仍缺乏系统研究。')]));

children.push(p([tr('在此背景下，本文提出一种面向部分生命周期监督的多尺度物理一致性'), tre('SOH'), tr('估计框架（'), tre('MS-PI-CNNLSTM'), tr('）。该方法的核心创新在于：（1）构建部分生命周期监督学习范式，在训练阶段保留完整充电特征轨迹，但仅对可用'), tre('SOH'), tr('标签区间计算数据监督损失，从而贴近实际'), tre('BMS'), tr('中健康标定稀疏、生命周期覆盖不完整的场景；（2）设计三路并行多尺度一维卷积模块（'), tre('kernel=3/7/15'), tr('），分别捕获充电过程中短程、中程与长程的电化学动态模式，为后续时序建模提供多粒度退化表征；（3）引入延迟激活的软单调物理一致性约束，使无标签样本虽不参与'), tre('MSE'), tr('监督，仍能通过退化单调性获得结构性弱监督信号；（4）通过系统消融分析揭示多尺度表征、物理约束与复杂模块堆叠之间的适用边界。')]));

children.push(p([tr('本文首先在标准'), tre('HUST'), tr('数据集上建立跨电池'), tre('SOH'), tr('估计基线，随后围绕部分生命周期监督设定，在不同监督比例（'), tre('r=1.0/0.7/0.5/0.3'), tr('）下系统评估架构设计与物理约束的各自贡献及其交互效应。实验与分析表明，多尺度卷积架构能够在有限标签条件下增强退化特征表达能力，而软单调约束在标签稀缺区间为未标注生命周期区间提供额外物理引导，二者共同构成一种面向实际'), tre('BMS'), tr('标定受限场景的可解释'), tre('SOH'), tr('估计路线。')]));

// ════════════════════════════════════════════════════════════════════════════
// 4.2 数据准备与特征工程
// ════════════════════════════════════════════════════════════════════════════
children.push(h2('4.2  数据准备与特征工程'));

// 4.2.1
children.push(h3('4.2.1  原始HUST数据集与特征空间构建'));

children.push(p([tr('本研究基于华中科技大学（'), tre('HUST'), tr('）公开的锂离子电池全生命周期退化数据集，选取其中77组'), tre('LFP/'), tr('石墨（'), tre('LFP/graphite'), tr('）电池单元作为研究对象。每个电池单元的标称容量为1.1 Ah，标称电压为3.3 V，所有样本在30 °C恒温环境下运行，采用统一的'), tre('CC-CV'), tr('充电协议，但在放电阶段施加了不同的多阶段放电工况，以模拟实际应用中负载变化带来的退化差异。')]));

children.push(p([tr('图4-1展示了该数据集中所有电池在全寿命循环过程中的容量（Ah）随循环次数变化的演化趋势。不同颜色曲线对应不同电池个体，可以观察到明显的个体差异与非线性衰减特征。整体上，电池容量随循环次数增加呈持续下降趋势，但在衰减速率和寿命终止点上表现出显著离散性，为模型在不同退化阶段的特征学习能力提供了充分的数据支撑。')]));
children.push(placeholder('【图4-1  HUST数据集中77节锂离子电池的容量衰减轨迹】'));
children.push(figcap('图4-1  HUST数据集中77节锂离子电池的容量衰减轨迹'));

children.push(p([tr('在特征工程方面，本文聚焦于恒流–恒压（'), tre('CC–CV'), tr('）充电过程中可稳定获取的外部可观测信号，从单个充电循环中提取统计特征与动力学特征相结合的多维特征向量。具体而言，围绕'), tre('CC'), tr('阶段的电压演化行为与'), tre('CV'), tr('阶段的电流衰减行为，分别构建反映信号幅值水平、波动特性、分布形态及时间演化规律的统计描述指标。')]));

children.push(p([tr('最终，从每个充电循环中共提取15个特征变量，包括电压与电流信号的均值、标准差、偏度、峰度，以及'), tre('CC'), tr('与'), tre('CV'), tr('阶段的充入电量和持续时间等关键动力学量，具体特征定义如表4-1所示。上述特征能够从不同角度刻画电池在退化过程中因活性物质损失、极化加剧及传输阻抗变化所引起的非线性响应差异，为后续'), tre('SOH'), tr('深度建模提供信息充分、结构稳定的输入空间。')]));

children.push(tablecap('表4-1  CC–CV充电循环中提取的统计特征与动力学特征定义'));
children.push(makeTable(
  ['特征编号', '特征名称', '物理含义', '所属阶段'],
  [
    ['F1', '电压均值', 'CC阶段端电压均值，反映平均极化水平', 'CC'],
    ['F2', '电压标准差', 'CC阶段电压波动幅度', 'CC'],
    ['F3', '电压峰度', 'CC阶段电压分布尖锐程度', 'CC'],
    ['F4', '电压偏度', 'CC阶段电压分布不对称性', 'CC'],
    ['F5', 'CC充入电量', 'CC段实际充入电荷量（Ah）', 'CC'],
    ['F6', 'CC充电时间', 'CC段持续时间（s）', 'CC'],
    ['F7', '电压斜率', 'CC段电压随时间的变化率', 'CC'],
    ['F8', '电压熵', 'CC段电压序列信息熵', 'CC'],
    ['F9', '电流均值', 'CV阶段充电电流均值', 'CV'],
    ['F10', '电流标准差', 'CV阶段电流波动幅度', 'CV'],
    ['F11', '电流峰度', 'CV阶段电流分布尖锐程度', 'CV'],
    ['F12', '电流偏度', 'CV阶段电流分布不对称性', 'CV'],
    ['F13', 'CV充入电量', 'CV段实际充入电荷量（Ah）', 'CV'],
    ['F14', 'CV充电时间', 'CV段持续时间（s）', 'CV'],
    ['F15', '电流斜率', 'CV段电流随时间的衰减率', 'CV'],
  ],
  [1000, 2000, 4300, 1546]
));

children.push(p([tr('为分析所构建特征集的信息互补性，图4-2给出了各特征变量之间及其与容量之间的'), tre('Pearson'), tr('相关系数矩阵。可以观察到，不同统计特征在一定程度上存在相关性，反映其对相同退化行为的不同侧面刻画；同时，多个特征与容量呈现出中等至较强的相关关系，表明所提取特征能够有效表征'), tre('SOH'), tr('的变化趋势。本文保留完整特征集合，由后续深度模型在训练过程中自动学习特征间的非线性组合关系。')]));
children.push(placeholder('【图4-2  HUST数据集中多源特征与容量退化之间的相关性矩阵】'));
children.push(figcap('图4-2  HUST数据集中多源特征与容量退化之间的相关性矩阵'));

// 4.2.2
children.push(h3('4.2.2  基于部分生命周期监督的退化样本构建'));

children.push(p([tr('在实际电池管理系统（'), tre('BMS'), tr('）应用中，'), tre('SOH'), tr('的标定通常依赖于周期性容量测试或离线诊断手段，受限于维护窗口、测试成本及运行条件，完整生命周期内连续、可靠的'), tre('SOH'), tr('标定信息往往难以获得。更为常见的情形是：不同电池仅在其服役生命周期的部分阶段具有可用的健康状态标定数据，而其余阶段处于"未观测"或"弱监督"状态。部分电池仅在服役初期完成容量标定，部分电池在中期或后期才被纳入监测体系，亦有电池仅在零散维护节点获得有限的健康评估结果，且已标定区段在生命周期维度上呈随机分布。')]));

children.push(p([tr('基于上述工程现实，本文不再假设每个电池样本均具有完整生命周期的'), tre('SOH'), tr('监督信息，而是将'), tre('SOH'), tr('估计问题建模为一种基于部分生命周期监督（'), tre('Partial Lifecycle Supervision'), tr('）的退化轨迹学习任务。在该设定下，训练阶段中不同电池样本仅暴露其生命周期中的部分区间作为监督信号，所覆盖的阶段在电池之间呈现明显差异：部分样本仅包含早期退化数据，部分样本覆盖中期或后期阶段，亦有样本跨越多个退化阶段但仍存在未观测区间。')]));

children.push(p([tr('形式化地，设第'), tre(' i '), tr('个电池的完整生命周期长度为'), tre('Tᵢ'), tr('，其充电循环特征序列与真实'), tre('SOH'), tr('演化轨迹分别表示为：')]));
children.push(formula(
  'X⁻ⁱ = {x⁻ⁱ₁, x⁻ⁱ₂, ..., x⁻ⁱₜᵢ},    y⁻ⁱ = {y⁻ⁱ₁, y⁻ⁱ₂, ..., y⁻ⁱₜᵢ}'
));

children.push(p([tr('在训练阶段，仅选取其中一个或多个子区间'), tre(' Ωᵢ ⊂ {1,...,Tᵢ} '), tr('作为可用'), tre('SOH'), tr('监督标签，其余时间步对应的'), tre('SOH'), tr('不参与数据误差项计算。为描述标签可用性，定义监督掩码：')]));
children.push(formula('m⁻ⁱₜ = 1,  if t ∈ Ωᵢ ;    m⁻ⁱₜ = 0,  if t ∉ Ωᵢ'));

children.push(p([tr('其中'), tre(' m⁻ⁱₜ=1 '), tr('表示第'), tre(' i '), tr('个电池第'), tre(' t '), tr('个循环具有'), tre('SOH'), tr('标签，'), tre('m⁻ⁱₜ=0 '), tr('表示该循环仅有运行特征而无'), tre('SOH'), tr('监督。对应的特征输入'), tre(' x⁻ⁱₜ '), tr('仍来源于完整充电循环过程，以保证模型在单个循环层面具备一致的输入结构。')]));

children.push(p([tr('需要强调的是，本文采用的是标签掩码（'), tre('Label Masking'), tr('）而非样本删除（'), tre('Sample Dropping'), tr('）策略。无标签循环不会被从训练集中删除，其运行特征仍参与模型前向传播；区别仅在于数据监督损失不对这些循环计算。该设定保留了完整生命周期特征轨迹，使模型能够在有标签区间学习特征–'), tre('SOH'), tr('映射，同时在无标签区间接受物理一致性约束的结构性引导。相较于直接丢弃无标签样本，'), tre('Label Masking'), tr('更符合实际'), tre('BMS'), tr('中"运行数据连续可得、'), tre('SOH'), tr('标定稀疏可得"的工程条件。')]));

children.push(p([tr('需要指出的是，在该设定下，模型在预测阶段仍遵循循环级别的一对一估计形式：即针对任意给定充电循环的特征输入，输出对应时刻的'), tre('SOH'), tr('预测值。部分生命周期监督仅作用于训练阶段的监督结构，而不改变模型在实际部署中的推理方式。此外，为避免模型通过隐式记忆特定电池的绝对退化位置进行投机性预测，本文采用严格的电池级别划分（'), tre('Battery-level Split'), tr('）策略，确保测试集中的电池个体在训练阶段完全不可见。')]));

children.push(placeholder('【图4-3  部分生命周期监督条件下训练电池的生命周期覆盖示意图（CRT数据集示例）】'));
children.push(figcap('图4-3  部分生命周期监督条件下训练电池的生命周期覆盖示意图'));

children.push(p([tr('图4-3展示了在部分生命周期监督设定下，不同训练电池在生命周期维度上的'), tre('SOH'), tr('标定覆盖情况。可以观察到，单个电池仅在其生命周期的部分区间内具有可用的'), tre('SOH'), tr('标签，而其余区间处于未观测状态；同时，不同电池的标定区间在生命周期维度上呈现明显差异，涵盖早期、中期及后期等不同退化阶段，且在单个电池生命周期内随机分布。该设定避免了对单一电池完整退化轨迹的依赖，更贴近实际'), tre('BMS'), tr('应用中健康状态标定稀疏且不连续的工程现实。在此条件下，模型需在不完整监督的约束下，仅依据局部退化特征学习合理的'), tre('SOH'), tr('映射关系，从而对其跨阶段泛化能力与物理一致性提出更高要求。')]));

// 4.2.3
children.push(h3('4.2.3  数据预处理与电池级验证划分'));

children.push(p([tr('为保证模型训练过程的稳定性与实验结论的科学性，本文在特征构建完成后，对数据执行了系统化的预处理与严格的样本划分策略。')]));

children.push(p([tr('首先，在特征层面采用3σ离群值检测准则对充电过程中的电压与电流特征进行异常值清洗。对于每一维特征，计算其均值与标准差，并剔除偏离均值超过3σ的异常采样点。为避免在复杂工况下降级处理过程中引入过度信息损失，本文额外设置了最小保留比例保护机制：若清洗后有效样本数量低于原始数据的80%，则回退使用未清洗数据，以确保统计稳定性。')]));

children.push(p([tr('随后，对全部特征向量采用'), tre('StandardScaler'), tr('标准化方法进行尺度对齐，并将目标'), tre('SOH'), tr('值线性映射至[0,1]区间，其中1表示初始健康状态，0表示达到寿命终止阈值。')]));

children.push(p([tr('在数据划分策略上，本文遵循电池级别划分（'), tre('Battery-level Split'), tr('）原则，而非传统的随机样本划分。具体而言，将77组电池样本按60%/20%/20%的比例划分为训练集（46组）、验证集（16组）与测试集（15组），并确保测试集中的电池个体在训练阶段完全不可见。该划分方式能够更真实地评估模型在面对未见电池个体与复杂工况退化数据时的泛化性能，相较于样本级随机划分更具工程意义。')]));

children.push(tablecap('表4-2  数据集划分与验证策略说明'));
children.push(makeTable(
  ['数据集', '电池数量', '样本比例', '用途'],
  [
    ['训练集', '46组', '60%', '模型参数学习'],
    ['验证集', '16组', '20%', '超参数调优与早停'],
    ['测试集', '15组', '20%', '跨电池泛化性能评估'],
  ],
  [2000, 2000, 2000, 2846]
));

// ════════════════════════════════════════════════════════════════════════════
// 4.3 架构设计
// ════════════════════════════════════════════════════════════════════════════
children.push(h2('4.3  MS-PI-CNNLSTM多尺度物理一致性神经网络架构设计'));

children.push(p([tr('在完成基于部分生命周期监督的特征构建与数据准备后，本文提出一种融合多尺度卷积特征提取与物理一致性约束的深度学习'), tre('SOH'), tr('估计框架'), tre('MS-PI-CNNLSTM'), tr('。该框架的设计逻辑并非简单堆叠网络模块，而是围绕部分生命周期监督任务中的两个核心矛盾展开：一方面，有限'), tre('SOH'), tr('标签要求模型具备更强的退化特征表达能力；另一方面，未标注生命周期区间缺少数据误差约束，需要物理先验提供结构性弱监督。基于此，'), tre('MS-PI-CNNLSTM'), tr('通过多尺度并行卷积模块增强对不同时间粒度退化特征的表征能力，同时引入软单调物理约束限制无标签区间的输出自由度，从而提升模型在标定受限场景下的稳定性与可解释性。')]));

// 4.3.1
children.push(h3('4.3.1  整体框架概述'));

children.push(p([tre('MS-PI-CNNLSTM'), tr('的整体架构由三个核心组件构成：（1）多尺度一维卷积特征提取模块，通过并行使用不同感受野的卷积核捕获充电过程中多粒度的电化学动态模式；（2）长短期记忆网络（'), tre('LSTM'), tr('）时序建模层，对卷积特征在循环维度上的长期退化趋势进行编码；（3）物理一致性弱监督模块，在损失函数层面嵌入退化单调性等物理先验，使未标注循环也能对模型训练产生约束作用。三者的分工关系为：多尺度'), tre('CNN'), tr('负责"看见"不同粒度的退化表征，'), tre('LSTM'), tr('负责"连接"跨循环的状态演化，物理约束负责在'), tre('SOH'), tr('标签缺失时提供"可行退化方向"。')]));

children.push(placeholder('【图4-4  MS-PI-CNNLSTM整体框架示意图（含多尺度并行卷积分支）】'));
children.push(figcap('图4-4  MS-PI-CNNLSTM整体框架示意图'));

// 4.3.2
children.push(h3('4.3.2  多尺度卷积特征提取模块'));

children.push(p([tr('锂离子电池在充放电循环过程中，其退化信号同时包含不同时间尺度的信息：短程尺度上，单次充电曲线中的电压/电流微观波动反映了即时电化学反应状态；中程尺度上，连续若干循环的特征变化趋势体现了活性物质损失与极化演化的阶段性特征；长程尺度上，数十至数百循环的整体演化规律揭示了电池的宏观退化路径。传统单尺度卷积网络使用固定大小的卷积核（如'), tre('kernel=7'), tr('），仅能在单一时间粒度上提取特征，难以同时捕获上述多层次的退化信息。')]));

children.push(p([tr('为此，本文设计了一种三路并行多尺度一维卷积模块（'), tre('Multi-Scale CNN Module'), tr('）。该模块使用三个独立的卷积分支，分别配置不同大小的卷积核，在相同的输入特征序列上并行提取多粒度的时序模式：')]));

children.push(listItem(1, '短程分支（kernel=3）：聚焦于相邻循环间的局部电化学响应变化，捕获电压曲线中的微观波动与高频动态特征。较小的感受野使其对单次充电行为的瞬态特性更为敏感，能够有效表征电池在短期应力下的即时响应差异。'));
children.push(listItem(2, '中程分支（kernel=7）：覆盖中等时间跨度的循环窗口，提取充电特征在连续若干循环中的过渡性变化模式。该尺度与传统单路CNN的典型配置相当，能够有效表征阶段性退化行为及容量衰减速率的中期变化。'));
children.push(listItem(3, '长程分支（kernel=15）：以更宽的感受野捕获跨越较多循环的宏观退化趋势，对整体容量衰减规律与慢变特征具有更强的表征能力，有助于建模电池从健康状态逐步进入加速退化阶段的长期演化轨迹。'));

children.push(p([tr('每个分支内部采用两层级联的"'), tre('Conv1d→BatchNorm→ReLU→MaxPool'), tr('"结构，通道数配置为[64, 64]，并使用'), tre('same-padding'), tr('保持序列长度一致。三个分支的输出在通道维度上进行拼接（'), tre('concatenation'), tr('），随后通过一个1×1卷积层融合至128维统一特征空间，作为后续'), tre('LSTM'), tr('层的输入。')]));

children.push(p([tr('与传统单路'), tre('CNN'), tr('（'), tre('channels=[256, 128], kernel=7'), tr('）相比，多尺度模块虽然总参数量相近，但通过"分而治之"的策略，能够在保持计算效率的同时显著提升特征的信息丰富度。在部分监督场景（'), tre('r=0.5'), tr('）下，该架构改进相较于单路基线带来了显著的'), tre('MAE'), tr('降低，是本文方法中收益最大且机制最为直接的单一结构创新。')]));

// 4.3.3
children.push(h3('4.3.3  LSTM时序建模与状态映射'));

children.push(p([tr('在多尺度卷积模块提取了循环内多粒度局部特征后，需要在循环维度上进一步建模'), tre('SOH'), tr('随时间演化的长期退化趋势。为此，本文采用双层堆叠的长短期记忆网络（'), tre('LSTM'), tr('）作为时序建模组件，隐状态维度设置为64。'), tre('LSTM'), tr('的门控机制能够在循环尺度上选择性地传递与更新状态信息，有助于隐式编码电池在不同退化阶段之间的时序依赖关系，从而将多尺度特征序列整合为具有历史感知能力的退化状态表示。')]));

children.push(p([tr('为缓解小样本条件下模型过拟合的问题，在'), tre('LSTM'), tr('层之间引入'), tre('Dropout'), tr('正则化策略，丢弃率通过验证集实验优化确定。在网络末端，采用全连接层结合'), tre('Sigmoid'), tr('激活函数，将高维隐层特征映射为归一化的'), tre('SOH'), tr('估计值。'), tre('Sigmoid'), tr('映射将模型输出约束在[0,1]区间内，使'), tre('SOH'), tr('估计结果始终位于容量退化的物理可行域中，同时为后续引入基于'), tre('SOH'), tr('单调退化先验的物理一致性约束提供连续可导的输出空间。')]));

// 4.3.4
children.push(h3('4.3.4  基于软单调性的物理一致性约束设计'));

children.push(p([tr('针对4.2节提出的部分生命周期监督学习任务，基础神经网络在缺乏完整生命周期先验的情况下，容易受到局部噪声与不完整监督信息的干扰，从而产生不符合电池退化物理规律的预测偏差。尤其是对于'), tre(' m⁻ⁱₜ=0 '), tr('的无标签循环，若仅依赖数据误差项训练，这些样本不会为模型提供任何'), tre('SOH'), tr('监督梯度。为此，本文引入一种基于软单调性（'), tre('Soft Monotonicity'), tr('）的物理一致性约束机制，使无标签循环也能通过退化方向先验参与训练。')]));

children.push(p([tr('该约束机制基于锂离子电池活性物质不可逆损耗的宏观电化学特性，要求模型预测的'), tre('SOH'), tr('演化轨迹在时间维度上整体呈非上升趋势。其物理约束损失项定义为：')]));
children.push(formula('Lₘₒₙₒ = (1/|P|) · Σ₊ᵢ₋ₜ₋ₜ₊₁₋ ∈ P  max(0,  ŷ⁻ⁱ₋ₜ₊₁₋  −  ŷ⁻ⁱₜ  −  ε)'));

children.push(p([tr('其中，'), tre('ŷ⁻ⁱₜ '), tr('与'), tre(' ŷ⁻ⁱ₋ₜ₊₁₋ '), tr('分别表示第'), tre(' i '), tr('个电池相邻循环的'), tre('SOH'), tr('预测值，'), tre('ε '), tr('为容忍因子（本文取'), tre('ε=0.005'), tr('），用于允许由测量噪声引起的轻微局部波动；'), tre('P '), tr('表示满足约束激活条件的相邻循环对集合。该软约束并不强制所有局部波动严格为零，而是仅惩罚超过容忍阈值的非物理上升，从而在物理一致性与真实数据噪声之间取得平衡。')]));

children.push(p([tr('在完整标签条件下，'), tre('Lₘₒₙₒ '), tr('可以被理解为对数据拟合项的辅助正则化；但在部分生命周期监督条件下，它具有更关键的机制意义：当某些循环缺少'), tre('SOH'), tr('标签时，数据损失'), tre(' Lₛₐₜₐ '), tr('不对这些样本计算，而'), tre(' Lₘₒₙₒ '), tr('仍可基于同一电池内相邻循环的预测关系提供梯度信号。因此，软单调约束不仅约束输出形态，也在无标签生命周期区间中承担结构性弱监督角色。这一点构成本文区别于普通物理正则化'), tre('SOH'), tr('模型的关键机制：物理约束在标签存在时是正则化手段，在标签缺失时是弱监督信号。')]));

children.push(p([tr('此外，考虑到锂离子电池在循环寿命早期可能由于活化过程、电极润湿改善或测试噪声等因素，出现短暂的容量回升或平台波动现象，图4-5展示了不同电池在初始循环阶段的'), tre('SOH'), tr('演化轨迹。可以观察到，在前若干循环内存在幅度有限但一致的容量上升趋势，随后各电池逐步进入稳定退化阶段，'), tre('SOH'), tr('呈现整体下降趋势。由此可见，若在全生命周期范围内无差别地施加单调性约束，可能对模型早期学习造成不必要的干扰。')]));
children.push(placeholder('【图4-5  锂离子电池在生命周期早期的容量回升现象示意图】'));
children.push(figcap('图4-5  锂离子电池在生命周期早期的容量回升现象示意图'));

children.push(p([tr('为此，本文进一步引入延迟激活的单调性约束机制，通过设定一个循环阈值（本文取'), tre(' min_cycle=300'), tr('），仅在循环索引'), tre(' i ≥ min_cycle '), tr('后对'), tre('SOH'), tr('预测轨迹施加软单调性约束，而在生命周期早期阶段不对容量回升行为进行强制惩罚。该设计充分尊重电池退化过程在不同寿命阶段的物理差异性，使模型在早期阶段能够灵活表征可能存在的非单调演化特征，而在进入稳定退化区间后，逐步引导'), tre('SOH'), tr('预测轨迹符合整体不可逆衰减的物理规律，从而在避免过度约束的同时提升长期预测的物理一致性与建模稳健性。')]));

// 4.3.5
children.push(h3('4.3.5  综合损失函数与评价指标'));

children.push(p([tre('MS-PI-CNNLSTM'), tr('的训练过程同时受到数据驱动误差项与物理一致性约束项的共同引导，其综合损失函数定义为：')]));
children.push(formula('Lₜₒₜₐₗ = Lₛₐₜₐ + λₘₒₙₒ · Lₘₒₙₒₜₒₙᵢᴄ'));

children.push(p([tr('其中'), tre(' Lₛₐₜₐ '), tr('为仅在有标签样本上计算的均方误差损失，定义为：')]));
children.push(formula('Lₛₐₜₐ = [ Σᵢ Σₜ m⁻ⁱₜ · (ŷ⁻ⁱₜ − y⁻ⁱₜ)² ] / [ Σᵢ Σₜ m⁻ⁱₜ ]'));

children.push(p([tr('其中'), tre(' m⁻ⁱₜ '), tr('为4.2.2节定义的监督掩码，'), tre('λₘₒₙₒ '), tr('为单调性约束权重（本文默认取0.3）。上述损失设计体现了本文方法的核心逻辑：有标签循环通过'), tre(' Lₛₐₜₐ '), tr('学习准确的'), tre('SOH'), tr('数值映射，无标签循环则通过'), tre(' Lₘₒₙₒₜₒₙᵢᴄ '), tr('参与退化方向约束。通过这种"数据监督+物理弱监督"的组合，模型能够在'), tre('SOH'), tr('标签不完整的条件下充分利用完整生命周期特征轨迹。')]));

children.push(p([tr('为定量评估所提出模型在'), tre('SOH'), tr('估计任务中的预测精度与误差特性，本文采用均方根误差（'), tre('RMSE'), tr('）、平均绝对误差（'), tre('MAE'), tr('）、决定系数（'), tre('R²'), tr('）以及单调违反率作为性能评价指标。其中'), tre('RMSE'), tr('对较大预测误差更加敏感，适用于评估模型在退化后期或复杂工况下对异常偏差的抑制能力；'), tre('MAE'), tr('反映预测误差的平均幅度；'), tre('R²'), tr('衡量模型的整体拟合质量；单调违反率用于评估预测轨迹对退化单调性物理规律的遵循程度，是衡量物理一致性的重要补充指标。')]));

// ════════════════════════════════════════════════════════════════════════════
// 4.4 实验
// ════════════════════════════════════════════════════════════════════════════
children.push(h2('4.4  实验结果与分析'));

children.push(p([tr('为系统评估所提出'), tre('MS-PI-CNNLSTM'), tr('模型在不同监督条件与工况复杂度下的'), tre('SOH'), tr('估计性能，本文围绕模型预测精度、架构贡献分析、物理弱监督机制与工程鲁棒性四个维度，设计并开展对比实验与分析。与单纯追求模块叠加不同，本节的实验组织服务于以下问题链：')]));

children.push(tablecap('表4-3  实验验证问题链与对应实验设计'));
children.push(makeTable(
  ['验证问题', '对应实验', '预期回答'],
  [
    ['部分生命周期监督是否构成独立挑战？', '不同监督比例r下的跨电池测试', '标签覆盖减少后模型是否仍能恢复完整SOH轨迹'],
    ['多尺度架构是否提升有限标签下的特征表达？', 'CNN-LSTM vs MS-CNN-LSTM', '多感受野卷积是否优于固定感受野卷积'],
    ['物理一致性约束是否为无标签区间提供弱监督？', '无物理 vs 软单调约束', '低标签比例下是否改善预测轨迹稳定性'],
    ['更多模块是否一定更优？', 'M1/M2/M4/M5/M6扩展模块消融', '复杂模块在部分监督下是否存在不稳定或冗余'],
  ],
  [2800, 2800, 3246]
));

// 4.4.1
children.push(h3('4.4.1  实验设置与对比方案'));

children.push(p([tr('所有实验均基于'), tre('HUST'), tr('数据集开展，严格遵循电池级别划分（'), tre('Battery-level split'), tr('）原则。模型输入为从单个充放电循环中提取的多维特征时序，模型输出为对应循环的'), tre('SOH'), tr('预测值。为系统分析架构设计与物理约束的各自贡献，本文设计如下主线对比方案：')]));

children.push(tablecap('表4-4  实验对比模型与配置说明'));
children.push(makeTable(
  ['模型简称', '架构', '物理约束', '说明'],
  [
    ['XGBoost', '梯度提升树', '无', '传统机器学习基线'],
    ['LSTM', '双层LSTM', '无', '纯时序建模基线'],
    ['CNN-LSTM (Eneg1)', '单路CNN+LSTM', '无', '深度学习无约束基线'],
    ['PI-CNN-LSTM (E0)', '单路CNN+LSTM', '软单调 w=0.3', '物理约束基线'],
    ['MS-CNN-LSTM (A1)', '多尺度CNN+LSTM', '无', '架构创新（纯结构对比）'],
    ['PI-MS-CNN-LSTM (Exp09c)', '多尺度CNN+LSTM', '软单调 w=0.3', '本文方法（最终方案）'],
  ],
  [2000, 2000, 2000, 2846]
));

children.push(p([tr('除上述主线模型外，本文将注意力机制（'), tre('M1'), tr('）、'), tre('MC Dropout'), tr('（'), tre('M2'), tr('）、速率连续性约束（'), tre('M4'), tr('）、自适应权重（'), tre('M5'), tr('）与不确定性伪标签（'), tre('M6'), tr('）作为扩展模块进行边界分析，用于回答"在部分生命周期监督场景下，哪些常见增强策略真正有效，哪些策略会因标签稀缺或校准不足而失效"。')]));

children.push(p([tr('所有深度学习模型均采用相同的训练配置：'), tre('Adam'), tr('优化器，'), tre('WarmupCosineDecay'), tr('学习率调度（'), tre('warmup 20 epochs'), tr('），'), tre('batch_size=1024'), tr('，早停策略（'), tre('patience=30'), tr('）。实验在不同随机种子下重复3次，结果取均值以验证稳定性。部分监督实验设置四种监督比例'), tre(' r∈{1.0, 0.7, 0.5, 0.3}'), tr('，分别模拟从完整标注到极度稀疏的不同工程场景。')]));

// 4.4.2
children.push(h3('4.4.2  全监督基准性能（r=1.0）'));

children.push(p([tr('在标准完整生命周期监督条件下（'), tre('r=1.0'), tr('），训练阶段为每个电池样本提供从初始状态至失效阶段的完整'), tre('SOH'), tr('演化标签。表4-5汇总了不同模型在测试集上的回归性能对比结果。')]));

children.push(tablecap('表4-5  全监督条件下（r=1.0）模型预测性能对比'));
children.push(makeTable(
  ['模型', 'RMSE (%)', 'MAE (%)', 'R²', '单调违反率 (%)'],
  [
    ['XGBoost', '—', '—', '—', '—'],
    ['LSTM', '—', '—', '—', '—'],
    ['CNN-LSTM (Eneg1)', '—', '—', '—', '—'],
    ['PI-CNN-LSTM (E0)', '—', '—', '—', '—'],
    ['MS-CNN-LSTM (A1)', '—', '—', '—', '—'],
    ['PI-MS-CNN-LSTM (Exp09c)', '—', '—', '—', '—'],
  ],
  [3000, 1500, 1500, 1300, 1546]
));

children.push(placeholder('【图4-6  全监督条件下各模型SOH预测散点图对比】'));
children.push(figcap('图4-6  全监督条件下各模型SOH预测散点图对比'));

children.push(p([tr('从整体指标上看，传统机器学习模型'), tre('XGBoost'), tr('能够给出相对稳定的预测，但在精度上明显落后于深度时序模型。时序神经网络模型在完整监督下展现出更强的退化规律建模能力，其中'), tre('CNN-LSTM'), tr('通过卷积层在循环内提取局部动态模式，并结合'), tre('LSTM'), tr('捕捉跨循环长期退化趋势，取得了较好的回归精度。')]));

children.push(p([tr('值得注意的是，全监督条件下（'), tre('r=1.0'), tr('）的实验不应被解读为本文创新性的主要来源。当训练数据充分覆盖完整退化轨迹时，模型能够直接从密集'), tre('SOH'), tr('标签中学习特征–'), tre('SOH'), tr('映射，单尺度'), tre('CNN-LSTM'), tr('已经具备较强的拟合能力，多尺度并行结构与物理约束的边际收益可能被完整标签所掩盖。因此，本文将全监督实验定位为基础性能验证，而非最终贡献证明。真正需要重点讨论的是：当'), tre('SOH'), tr('标签只覆盖生命周期的一部分时，多尺度表征和物理一致性约束是否能够弥补标签缺失带来的学习困难。')]));

// 4.4.3
children.push(h3('4.4.3  架构消融实验：多尺度特征提取的贡献'));

children.push(p([tr('为定量分析架构改进与物理约束各自的贡献及其交互效应，本文设计了一组2×3交叉对比实验，在两种架构（单路'), tre('CNN-LSTM vs'), tr('多尺度'), tre('MS-CNN-LSTM'), tr('）与三种约束级别（无物理约束、仅软单调约束、含'), tre('M2+M4'), tr('模块的'), tre('Full Stack'), tr('）下进行系统对比。该矩阵的意义在于将"特征表达能力"与"物理约束强度"解耦：若行方向改善显著，说明多尺度表征是主要贡献；若列方向改善显著，说明物理约束或附加模块更关键。实验覆盖四种监督比例（'), tre('r=1.0/0.7/0.5/0.3'), tr('），结果如表4-6所示。')]));

children.push(tablecap('表4-6  架构×物理约束 2×3 交叉对比（MAE%，3 seeds 均值）'));
children.push(makeTable(
  ['架构', '无物理约束', '仅软单调约束', 'Full Stack (M2+M4)'],
  [
    ['CNN-LSTM', '—', '—', '—'],
    ['MS-CNN-LSTM', '—', '—', '—'],
  ],
  [2846, 2000, 2000, 2000]
));

children.push(tablecap('表4-7  纯架构对比（无物理约束）：MS-CNN-LSTM（A1）vs CNN-LSTM（Eneg1）——各监督比例下MAE详细结果'));
children.push(makeTable(
  ['监督比例', 'CNN-LSTM MAE(%)', 'MS-CNN-LSTM MAE(%)', '改善幅度'],
  [
    ['r=1.0', '—', '—', '—'],
    ['r=0.7', '—', '—', '—'],
    ['r=0.5', '—', '—', '—'],
    ['r=0.3', '—', '—', '—'],
  ],
  [2000, 2200, 2200, 2446]
));

children.push(tablecap('表4-8  同等软单调约束下架构对比：PI-MS-CNN-LSTM（Exp09c）vs PI-CNN-LSTM（E0）'));
children.push(makeTable(
  ['监督比例', 'PI-CNN-LSTM MAE(%)', 'PI-MS-CNN-LSTM MAE(%)', '改善幅度'],
  [
    ['r=1.0', '—', '—', '—'],
    ['r=0.7', '—', '—', '—'],
    ['r=0.5', '—', '—', '—'],
    ['r=0.3', '—', '—', '—'],
  ],
  [2000, 2200, 2400, 2246]
));

children.push(p([tr('上述结果揭示了以下关键发现：')]));

children.push(p([tr('（1）'), tr('多尺度架构在部分监督场景下收益显著', true), tr('。在无物理约束条件下，'), tre('MS-CNN-LSTM'), tr('在'), tre('r=0.5'), tr('时'), tre('MAE'), tr('出现明显降低，表明多尺度并行卷积通过同时捕获不同时间粒度的退化特征，能够在训练数据受限时更高效地利用有限信息构建退化映射。')]));

children.push(p([tr('（2）'), tr('同等物理约束下，多尺度架构全面优于单路架构', true), tr('。'), tre('PI-MS-CNN-LSTM'), tr('在所有监督比例下均优于'), tre('PI-CNN-LSTM'), tr('，进一步证实了架构改进的普适性。')]));

children.push(p([tr('（3）'), tr('架构维度的改善大于约束维度', true), tr('。对比2×3矩阵可以发现，行方向（架构更换）的改善幅度整体大于列方向（约束增强），说明在'), tre('SOH'), tr('估计任务中，特征提取能力的提升比显式物理约束更为关键。')]));

children.push(p([tr('这一结果具有重要的论文叙事价值：它说明物理一致性约束并不能替代有效的退化表征学习。对于'), tre('HUST'), tr('这类跨电池数据集，不同电池在寿命长度、局部噪声、退化斜率和容量回升阶段上存在明显差异，若底层特征提取器只能观察单一时间尺度，则物理约束只能在输出端限制非物理解，而无法补偿输入表征不足。多尺度'), tre('CNN'), tr('的作用在于先获得更丰富的退化表征，再由'), tre('LSTM'), tr('和软单调约束对这些表征进行时序组织和物理修正。因此，本文方法的合理解释是"多尺度表征提供主要建模能力，物理约束在标签稀缺区间补充结构性引导"，而非"物理约束直接带来全部性能提升"。')]));

children.push(placeholder('【图4-7  2×3对比矩阵热力图（MAE%，架构×约束×监督比例）】'));
children.push(figcap('图4-7  架构与物理约束交互效应热力图'));

// 4.4.4
children.push(h3('4.4.4  部分监督鲁棒性评估'));

children.push(p([tr('为更贴近实际'), tre('BMS'), tr('应用场景，本文在训练阶段采用部分生命周期监督设定，系统评估不同模型在四种监督比例（'), tre('r=1.0/0.7/0.5/0.3'), tr('）下的性能变化。表4-9汇总了四种主要模型在各监督比例下的'), tre('MAE'), tr('对比结果。')]));

children.push(tablecap('表4-9  不同监督比例下各模型MAE（%）对比（seeds均值）'));
children.push(makeTable(
  ['模型', 'r=1.0', 'r=0.7', 'r=0.5', 'r=0.3'],
  [
    ['CNN-LSTM (Eneg1)', '—', '—', '—', '—'],
    ['PI-CNN-LSTM (E0)', '—', '—', '—', '—'],
    ['MS-CNN-LSTM (A1)', '—', '—', '—', '—'],
    ['PI-MS-CNN-LSTM (Exp09c)', '—', '—', '—', '—'],
  ],
  [3200, 1400, 1400, 1400, 1446]
));

children.push(placeholder('【图4-8  不同监督比例下各模型MAE变化趋势图（折线图）】'));
children.push(figcap('图4-8  不同监督比例下各模型MAE变化趋势对比'));

children.push(p([tr('从上述结果可以得出以下关键发现：')]));

children.push(p([tr('（1）'), tr('多尺度架构在中等监督条件下达到全场最优精度', true), tr('。'), tre('MS-CNN-LSTM'), tr('（'), tre('A1'), tr('）在'), tre('r=0.5'), tr('时'), tre('MAE'), tr('达到最低值，为所有模型–'), tre('ratio'), tr('组合中的最优结果。这表明多尺度卷积结构在中等监督条件下能够最有效地从有限标注数据中提取多粒度退化特征，相较于标签充足（'), tre('r=1.0'), tr('）时边际收益更为显著。')]));

children.push(p([tr('（2）'), tr('物理约束在低标签率下为多尺度架构提供正则化增益', true), tr('。'), tre('PI-MS-CNN-LSTM'), tr('（'), tre('Exp09c'), tr('）在'), tre('r=0.3'), tr('时优于无约束的'), tre('MS-CNN-LSTM'), tr('，'), tre('R²'), tr('也有所提升。这说明当标签极度稀缺时，软单调性约束作为一种结构性正则化手段，能够有效限制模型在未观测区间内的预测自由度，防止出现不合理的容量回弹现象。')]));

children.push(p([tr('（3）'), tr('物理约束在高监督条件下可能拖累精度', true), tr('。在'), tre('r=1.0'), tr('和'), tre('r=0.5'), tr('时，'), tre('PI-MS-CNN-LSTM'), tr('的'), tre('MAE'), tr('均高于无约束的'), tre('MS-CNN-LSTM'), tr('。这表明当训练数据充足时，多尺度架构已隐式学习到退化的物理规律，额外的显式约束反而限制了模型对个体差异和局部波动的拟合能力。')]));

children.push(p([tr('（4）上述发现具有重要的工程启示：在实际'), tre('BMS'), tr('部署中，应根据可获得的'), tre('SOH'), tr('标定信息密度灵活选择是否启用物理约束——标签充足时以纯架构模型为主，标签稀缺时加入轻量物理约束进行正则化。')]));

children.push(p([tr('由此可将物理一致性约束的作用边界概括为：当'), tre('SOH'), tr('标签较为充足时，数据监督本身已经携带充分的退化趋势信息，此时显式单调约束可能限制模型对局部容量回升、测试噪声或个体差异的拟合能力；当'), tre('SOH'), tr('标签极度稀缺时，未标注循环无法通过'), tre('MSE'), tr('获得训练信号，软单调约束则能够提供最基本的退化方向约束，减少输出空间中的非物理解。因此，物理约束的工程价值并非"所有场景都降低'), tre('MAE'), tr('"，而是"在标定不足时提高预测轨迹的可行性和稳定性"。')]));

children.push(tablecap('表4-10  典型监督比例下各方法综合性能对比'));
children.push(makeTable(
  ['指标', 'CNN-LSTM (r=0.5)', 'PI-MS-CNN-LSTM (r=0.5)', 'PI-MS-CNN-LSTM (r=0.3)'],
  [
    ['MAE (%)', '—', '—', '—'],
    ['RMSE (%)', '—', '—', '—'],
    ['R²', '—', '—', '—'],
    ['单调违反率 (%)', '—', '—', '—'],
  ],
  [2800, 2000, 2000, 2046]
));

children.push(placeholder('【图4-9  未见电池上的SOH预测轨迹对比（r=0.3，典型两块电池）】'));
children.push(figcap('图4-9  未见电池上的SOH预测轨迹对比（r=0.3）'));

children.push(p([tr('图4-9进一步给出了两块未见电池上的'), tre('SOH'), tr('预测轨迹对比结果。可以观察到，'), tre('CNN-LSTM'), tr('在未标注区间内更容易出现局部震荡或阶段性偏移，而'), tre('PI-MS-CNN-LSTM'), tr('的预测轨迹整体更加平滑、连续，能够在完整生命周期范围内保持符合退化直觉的单调演化趋势，尤其在退化后期阶段对容量加速衰减阶段的捕捉更为稳定。')]));

// 4.4.5
children.push(h3('4.4.5  扩展模块的收益边界与复杂度分析'));

children.push(p([tr('除多尺度架构与软单调约束外，本文还围绕注意力机制、不确定性估计、速率连续性约束、自适应损失权重和伪标签训练等扩展模块进行了探索。与其将这些模块简单堆叠进最终模型，本文更关注它们在部分生命周期监督场景下的适用边界。表4-11给出了各扩展模块的机制定位与论文处理建议。')]));

children.push(tablecap('表4-11  扩展模块的机制定位与边界分析'));
children.push(makeTable(
  ['模块', '设计初衷', '部分监督下的潜在风险', '论文定位建议'],
  [
    ['M1 循环级注意力', '在窗口内突出关键循环特征', '低标签下可能过拟合少量有标签片段，r=0.5时单调违反率飙升至基线2.5×', '作为消融项，不作为核心贡献'],
    ['M2 MC Dropout', '估计预测不确定性，提供置信区间', 'Dropout方差未必等价于真实误差，点估计受采样噪声影响', '作为不确定性辅助分析'],
    ['M4 速率连续性', '抑制预测斜率突变', '真实退化存在阶段性加速，过强平滑会压制有效变化', '作为物理扩展约束，报告边界'],
    ['M5 自适应权重', '自动平衡数据损失与物理损失', '可学习权重不直接感知监督比例，低标签下可能关闭关键约束（全监督下MAE退化12%）', '不进主线，作负例分析'],
    ['M6 伪标签', '为无标签循环补充伪SOH监督', '不确定性校准不足时，高置信伪标签可能系统性错误，r=0.3下单调违反率达18%±22%', '不进主线，作限制讨论'],
  ],
  [1200, 2000, 2800, 2846]
));

children.push(p([tr('上述分析说明，在部分生命周期监督任务中，"更多模块"并不必然带来更可靠的'), tre('SOH'), tr('估计。注意力、伪标签和自适应权重等方法都依赖更强的训练信号或更可靠的校准条件；当'), tre('SOH'), tr('标签本身稀缺时，这些模块可能放大局部偏差或引入额外不稳定性。相比之下，多尺度'), tre('CNN'), tr('和软单调约束分别对应"更强表征"与"更明确先验"，与部分监督问题的核心矛盾直接匹配，因此更适合作为最终论文主线。')]));

children.push(p([tr('具体地，'), tre('M1'), tr('注意力机制在全监督条件下（'), tre('r=1.0'), tr('）贡献几乎为零（'), tre('MAE'), tr('差异低于噪声水平），而在'), tre('r=0.5'), tr('下'), tre('MAE'), tr('出现退化、单调违反率显著上升，说明注意力权重在标签稀疏时倾向于过拟合少量有标签循环的局部模式，从而破坏全局单调先验。'), tre('M5'), tr('自适应权重在全监督下表现最差，根因在于可学习的'), tre('log_var'), tr('参数在训练中将单调性权重持续压低，导致物理约束实际上被关闭，梯度信号严重失衡。'), tre('M6'), tr('伪标签在'), tre('r=0.3'), tr('场景下存在"负偏差自我强化"问题：初始轻微低估导致伪标签偏低，模型学偏后下一轮伪标签更偏，形成放大循环，seed间方差高达22.7%。')]));

children.push(p([tr('因此，本文最终不把模型贡献表述为多模块堆叠，而是强调一种更具工程可解释性的简洁路线：先通过多尺度卷积提高有限标签条件下的退化表征能力，再用轻量物理一致性约束为无标签区间提供退化方向弱监督。该路线具有更低实现复杂度、更清晰机制解释和更好的工程可迁移性。')]));

// 4.4.6
children.push(h3('4.4.6  未见电池泛化能力分析'));

children.push(p([tr('为进一步验证多尺度架构与物理一致性约束对于提升模型跨样本泛化能力的贡献，本节选取测试集中若干具有代表性的未见电池进行对比分析。')]));

children.push(placeholder('【图4-10  未见电池样本上的SOH预测轨迹对比（多模型）】'));
children.push(figcap('图4-10  未见电池样本上的SOH预测轨迹对比'));

children.push(p([tr('如图4-10所示，在未参与训练的电池样本上，标准'), tre('CNN-LSTM'), tr('模型虽然能够在一定程度上跟随'), tre('SOH'), tr('随循环数下降的总体退化趋势，但当面对不同电池个体在退化速率、噪声水平或局部工况变化方面的差异时，其预测结果在局部时间段内仍可能出现不稳定行为，尤其在退化中后期容量加速衰减阶段误差增大。')]));

children.push(p([tr('相比之下，'), tre('MS-CNN-LSTM'), tr('通过多尺度特征提取获得了更为鲁棒的退化表征，在不同未见电池上均表现出更平滑的预测轨迹。进一步引入物理约束的'), tre('PI-MS-CNN-LSTM'), tr('则在此基础上额外抑制了零星的非物理波动，尤其在退化后期阶段保持了更为一致的单调下降趋势，体现了多尺度特征表达与物理弱监督的协同作用。')]));

children.push(placeholder('【图4-11  未见电池样本上的预测误差演化曲线对比】'));
children.push(figcap('图4-11  未见电池样本上的预测误差演化曲线对比'));

children.push(p([tr('如图4-11所示，'), tre('PI-MS-CNN-LSTM'), tr('的误差曲线在各测试电池上均表现得更加平稳且接近零轴，说明多尺度架构与物理约束的协同作用不仅提升了预测精度，还增强了模型在面对未见样本时的输出一致性，尤其是对于退化速率与训练集差异较大的电池个体，其预测轨迹的物理合理性得到了有效保证。')]));

// ════════════════════════════════════════════════════════════════════════════
// 4.5 本章小结
// ════════════════════════════════════════════════════════════════════════════
children.push(h2('4.5  本章小结'));

children.push(p([tr('本章围绕实际'), tre('BMS'), tr('中'), tre('SOH'), tr('标定不完整、生命周期标签覆盖有限的工程问题，系统研究了一种面向部分生命周期监督的多尺度物理一致性'), tre('SOH'), tr('估计方法'), tre('MS-PI-CNNLSTM'), tr('。与传统完整生命周期监督假设不同，本文将运行特征连续可得而'), tre('SOH'), tr('标签稀疏可得的场景显式建模为标签掩码（'), tre('Label Masking'), tr('）问题，从而更贴近电池实际服役过程中的数据获取条件。')]));

children.push(p([tr('在方法设计方面，本章的核心创新包括两个层面。首先，设计多尺度并行卷积特征提取模块，通过三路不同感受野的卷积分支（'), tre('kernel=3/7/15'), tr('）同时捕获充电过程中短程、中程与长程的电化学动态模式，为后续时序建模提供信息更为丰富的多粒度退化表示。其次，在标签掩码监督协议下引入延迟激活的软单调物理一致性约束，使无标签生命周期区间虽然不参与数据误差项计算，仍能够通过退化方向先验获得结构性弱监督信号。')]));

children.push(p([tr('通过系统的消融实验与机制分析，本章形成了以下关键结论：')]));

children.push(p([tr('（1）'), tr('部分生命周期监督是本文的核心问题设定', true), tr('。该设定不删除无标签循环，而是保留完整运行特征轨迹，仅屏蔽缺失'), tre('SOH'), tr('标签的数据误差项。由此，模型既面临有限标签下的跨阶段泛化问题，也需要在无标签区间获得额外结构性约束。')]));

children.push(p([tr('（2）'), tr('多尺度架构是主要表征贡献', true), tr('。电池退化信号同时包含短程局部波动、中程阶段变化与长程衰退趋势，单一卷积感受野难以充分刻画这种多粒度结构。多尺度'), tre('CNN'), tr('通过并行感受野增强有限标签条件下的退化特征表达能力，是本文方法中最直接、最稳定、最具工程解释性的结构改进，在中等标签密度场景下效果尤为突出。')]));

children.push(p([tr('（3）'), tr('物理一致性约束的价值在于结构性弱监督，而非无条件提升点估计精度', true), tr('。当'), tre('SOH'), tr('标签较为充足时，数据监督本身已经包含较完整的退化趋势信息，显式物理约束可能限制模型对个体差异和局部波动的拟合；当标签极度稀缺时，软单调约束能够为无标签区间提供退化方向引导，缩小输出空间中的非物理解集合，提高预测轨迹的物理可行性。')]));

children.push(p([tr('（4）'), tr('复杂模块存在适用边界，简洁机制更利于工程部署', true), tr('。注意力机制、'), tre('MC Dropout'), tr('、自适应权重和伪标签扩展均具有合理动机，但在部分生命周期监督场景下可能受到低标签、校准不足或权重漂移影响。本文最终强调"多尺度表征+轻量物理弱监督"的主线，不以模块数量衡量方法复杂度。')]));

children.push(p([tr('综上所述，本章提出的'), tre('MS-PI-CNNLSTM'), tr('并非单纯追求网络复杂度，而是围绕真实'), tre('BMS'), tr('中'), tre('SOH'), tr('标签不完整这一工程痛点，构建了一条可解释的建模路线：以多尺度卷积提升有限标签下的退化特征表达，以'), tre('LSTM'), tr('建模跨循环状态演化，以软单调约束为无标签生命周期区间提供结构性弱监督。该路线兼具问题针对性、机制解释性与工程可实现性，为标定受限条件下的锂离子电池'), tre('SOH'), tr('估计提供了可推广的方法框架。')]));

// ═══════════════════════════════════════════════════════════════════════════════
// 生成文档
// ═══════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: { name: FONT_CN, eastAsia: FONT_CN }, size: SZ_BODY }
      }
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SZ_H1, bold: true, font: { name: FONT_CN, eastAsia: FONT_CN } },
        paragraph: {
          spacing: { line: 360, lineRule: 'auto', before: 480, after: 240 },
          alignment: AlignmentType.CENTER, outlineLevel: 0
        }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SZ_H2, bold: true, font: { name: FONT_CN, eastAsia: FONT_CN } },
        paragraph: {
          spacing: { line: 360, lineRule: 'auto', before: 360, after: 120 },
          alignment: AlignmentType.LEFT, outlineLevel: 1
        }
      },
      {
        id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SZ_H3, bold: true, font: { name: FONT_CN, eastAsia: FONT_CN } },
        paragraph: {
          spacing: { line: 360, lineRule: 'auto', before: 240, after: 60 },
          alignment: AlignmentType.LEFT, outlineLevel: 2
        }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: A4_W, height: A4_H },
        margin: MARGIN
      }
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('D:\\Projects\\1111-soh\\第四章_正文.docx', buf);
  console.log('✅ 已生成：D:\\Projects\\1111-soh\\第四章_正文.docx');
}).catch(e => { console.error(e); process.exit(1); });
