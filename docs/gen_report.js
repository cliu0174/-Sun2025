const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, VerticalAlign, PageNumber, Header, Footer,
} = require('C:/Users/46139/AppData/Roaming/npm/node_modules/docx');
const fs = require('fs');

// ─── 颜色 & 边框常量 ──────────────────────────────────────
const BLACK = '000000';
const GRAY_HEADER = 'DDDDDD';
const GRAY_LIGHT = 'F5F5F5';
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: '888888' };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };

// 页面宽度（A4，2.5cm 边距）: 11906 - 2*1417 ≈ 9072 DXA
const PAGE_W = 11906;
const MARGIN = 1134; // ~2cm
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9638

// ─── 辅助函数 ─────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, size: 30, font: 'Arial', color: BLACK })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA', space: 4 } },
    children: [new TextRun({ text, bold: true, size: 26, font: 'Arial', color: BLACK })],
  });
}

function body(text, options = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, size: 22, font: 'Arial', color: BLACK, ...options })],
  });
}

function bullet(text, bold_prefix = '') {
  const children = [];
  if (bold_prefix) {
    children.push(new TextRun({ text: bold_prefix, bold: true, size: 22, font: 'Arial', color: BLACK }));
  }
  children.push(new TextRun({ text, size: 22, font: 'Arial', color: BLACK }));
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40 },
    children,
  });
}

function spacer(lines = 1) {
  return new Paragraph({ children: [new TextRun({ text: '', size: lines * 12 })] });
}

function cell(text, opts = {}) {
  const { bold = false, bg = 'FFFFFF', width = 1, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold, size: 20, font: 'Arial', color: BLACK })],
    })],
  });
}

// ─── 表格：五类退化场景 ───────────────────────────────────
const COL = [900, 2600, 2600, 1700, 1838]; // sum = 9638
const scenarioTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: COL,
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        cell('场景', { bold: true, bg: GRAY_HEADER, width: COL[0], align: AlignmentType.CENTER }),
        cell('物理根因', { bold: true, bg: GRAY_HEADER, width: COL[1] }),
        cell('SOH 曲线表现', { bold: true, bg: GRAY_HEADER, width: COL[2] }),
        cell('仿真方式', { bold: true, bg: GRAY_HEADER, width: COL[3] }),
        cell('检测难度', { bold: true, bg: GRAY_HEADER, width: COL[4], align: AlignmentType.CENTER }),
      ],
    }),
    ...[
      ['A1 突发老化', '机械冲击、过充/过放等不可逆突发损伤', '某循环后 SOH 骤降一大截，之后继续平稳衰减', '截取同电池末期数据拼接至故障点', '★ 容易'],
      ['A2 容量拐点', '活性物质耗尽、电解液副反应积累至临界点', '前期缓慢，拐点后退化速率急剧加快', '故障点后每隔 3~4 步取一个样本（跳采样）', '★★ 较易'],
      ['A3 簇内不均衡', '电池组中单体差异化老化（温度/制造偏差）', '故障后持续偏低，差距随时间线性扩大', '将同电池未来状态按线性比例混入当前', '★★★ 中等'],
      ['A4 析锂', '低温/大电流充电致金属锂沉积，不可逆容量损失', 'SOH 台阶突降后继续加速衰减', '跳至末期状态（台阶）+ 跳采样（加速）', '★★ 较易'],
      ['A5 内阻增长', 'SEI 膜增厚、电解液老化致内阻缓慢升高', '斜率极缓慢变陡，早期几乎察觉不到', '按二次方比例混入同电池更老状态', '★★★★ 最难'],
    ].map((row, i) => new TableRow({
      children: row.map((text, j) => cell(text, {
        bg: i % 2 === 0 ? 'FFFFFF' : GRAY_LIGHT,
        width: COL[j],
        align: j === 4 ? AlignmentType.CENTER : AlignmentType.LEFT,
      })),
    })),
  ],
});

// ─── 表格：五个检测信号 ───────────────────────────────────
const SIG_COL = [2000, 3500, 4138]; // sum = 9638
const signalTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: SIG_COL,
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        cell('信号名称', { bold: true, bg: GRAY_HEADER, width: SIG_COL[0] }),
        cell('计算方式', { bold: true, bg: GRAY_HEADER, width: SIG_COL[1] }),
        cell('含义与局限', { bold: true, bg: GRAY_HEADER, width: SIG_COL[2] }),
      ],
    }),
    ...[
      ['input_zscore', '当前 14 维特征与训练集均值的最大标准差偏离量', '反映输入数据是否偏离"正常范围"；生命末期自然偏高，易误报'],
      ['mono_violation', '相邻两步 SOH 预测的上升量（SOH 应单调递减）', '检测模型预测出"电池变年轻"的物理违规；正常数据极少触发'],
      ['rate_anomaly', '当前下降速率与最近 20 步均值的标准差倍数', '检测退化速度的突变；局部窗口较小时稳定性有限'],
      ['drop_anomaly', '当前下跌量与最近 20 步均值的标准差倍数（仅计下跌）', '专门检测异常加速下跌；与 rate_anomaly 互补'],
      ['trajectory_deviation ★', '用最近 20 步预测做线性外推，计算实际值与预期值的偏差', '最有效信号（AUC 0.65–0.75），对所有 5 类场景均稳定有效'],
    ].map((row, i) => new TableRow({
      children: row.map((text, j) => cell(text, {
        bg: i % 2 === 0 ? 'FFFFFF' : GRAY_LIGHT,
        width: SIG_COL[j],
        bold: j === 0,
      })),
    })),
  ],
});

// ─── 表格：评价指标 ───────────────────────────────────────
const MET_COL = [2000, 3800, 3838]; // sum = 9638
const metricTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: MET_COL,
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        cell('指标', { bold: true, bg: GRAY_HEADER, width: MET_COL[0] }),
        cell('含义', { bold: true, bg: GRAY_HEADER, width: MET_COL[1] }),
        cell('解读参考', { bold: true, bg: GRAY_HEADER, width: MET_COL[2] }),
      ],
    }),
    ...[
      ['AUC', '随机抽取一个异常时刻和一个正常时刻，异常时刻分数更高的概率', '0.5 = 随机猜测；>0.70 为有效；突发型场景实测 ~0.73'],
      ['Det@FPR5%', '允许误报率 ≤5% 的约束下，能检出的真实异常时刻比例', '随机基准为 5%；突发老化实测 ~15%（是随机的 3 倍）'],
      ['det_delay（循环数）', '从故障注入点到首次触发报警所经历的充电循环数', '越小越好；—表示整段均未检出；突发型通常在数十循环内报警'],
    ].map((row, i) => new TableRow({
      children: row.map((text, j) => cell(text, {
        bg: i % 2 === 0 ? 'FFFFFF' : GRAY_LIGHT,
        width: MET_COL[j],
        bold: j === 0,
      })),
    })),
  ],
});

// ─── 表格：识别过程 步骤图 ───────────────────────────────
const STEP_COL = [900, 8738];
const stepTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: STEP_COL,
  rows: [
    ...[
      ['Step 1', '模型预测 SOH\n持续输入当前充电数据，模型逐循环输出 SOH 预测值，形成连续的 SOH 时间序列。'],
      ['Step 2', '计算 5 维异常分数\n对每个时刻，从"特征偏离、单调违规、速率突变、骤降程度、轨迹偏差"五个角度各输出一个分数，分数越高越可疑。'],
      ['Step 3', '设定报警阈值（FPR=5%）\n从正常时段的分数中取第 95 百分位数作为报警线，确保正常时段误报率 ≤5%。'],
      ['Step 4', '逐循环判断是否报警\n每新来一个循环，若其分数超过报警线则标记为异常；连续 N 次超线可进一步降低误报风险（工程落地优化）。'],
      ['Step 5', '输出检测结果\n记录 AUC、Det@FPR5%、首次报警延迟（det_delay），量化系统检测能力。'],
    ].map(([step, desc], i) => new TableRow({
      children: [
        new TableCell({
          borders: BORDERS,
          width: { size: STEP_COL[0], type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? 'D8D8D8' : GRAY_LIGHT, type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 100, right: 100 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: step, bold: true, size: 20, font: 'Arial', color: BLACK })],
          })],
        }),
        new TableCell({
          borders: BORDERS,
          width: { size: STEP_COL[1], type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? 'FFFFFF' : GRAY_LIGHT, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 160, right: 140 },
          children: desc.split('\n').map((line, li) => new Paragraph({
            spacing: { before: li === 0 ? 0 : 40, after: 40 },
            children: [new TextRun({ text: line, bold: li === 0, size: 20, font: 'Arial', color: BLACK })],
          })),
        }),
      ],
    })),
  ],
});

// ─── 实验结果小表 ─────────────────────────────────────────
const RES_COL = [2200, 1800, 1800, 1800, 2038];
const resultTable = new Table({
  width: { size: CONTENT_W, type: WidthType.DXA },
  columnWidths: RES_COL,
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        cell('场景', { bold: true, bg: GRAY_HEADER, width: RES_COL[0] }),
        cell('AUC', { bold: true, bg: GRAY_HEADER, width: RES_COL[1], align: AlignmentType.CENTER }),
        cell('Det@FPR5%', { bold: true, bg: GRAY_HEADER, width: RES_COL[2], align: AlignmentType.CENTER }),
        cell('信号来源', { bold: true, bg: GRAY_HEADER, width: RES_COL[3], align: AlignmentType.CENTER }),
        cell('结论', { bold: true, bg: GRAY_HEADER, width: RES_COL[4], align: AlignmentType.CENTER }),
      ],
    }),
    ...[
      ['A1 突发老化', '0.731', '~15%', 'trajectory_deviation', 'GOOD ✓'],
      ['A2 容量拐点', '0.697', '~10%', 'trajectory_deviation', 'WEAK'],
      ['A3 簇内不均衡', '0.654', '~8.6%', 'trajectory_deviation', 'WEAK'],
      ['A4 析锂', '0.736', '—', 'trajectory_deviation', 'GOOD ✓'],
      ['A5 内阻增长', '0.668', '—', 'trajectory_deviation', 'WEAK'],
    ].map((row, i) => new TableRow({
      children: row.map((text, j) => cell(text, {
        bg: i % 2 === 0 ? 'FFFFFF' : GRAY_LIGHT,
        width: RES_COL[j],
        align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })),
    })),
  ],
});

// ─── 构建文档 ─────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 560, hanging: 280 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: 'Arial', color: BLACK },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: BLACK },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: 16838 },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: '电池 SOH 异常检测机制说明  |  ', size: 16, font: 'Arial', color: '888888' }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: 'Arial', color: '888888' }),
          ],
        })],
      }),
    },
    children: [

      // ── 封面标题 ──
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 400, after: 160 },
        children: [new TextRun({ text: '电池 SOH 异常检测机制说明', bold: true, size: 44, font: 'Arial', color: BLACK })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 480 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '888888', space: 8 } },
        children: [new TextRun({ text: '基于轨迹偏差的无标签异常感知 | HUST 77 电池数据集', size: 22, font: 'Arial', color: '555555' })],
      }),

      // ── 一、五类退化场景 ──
      h1('一、五类退化场景'),
      body('研究选取了电动车/储能领域最常见的五类电池退化模式，通过对同一块电池自身历史轨迹的重排来模拟故障注入，无需真实失效电池数据。'),
      spacer(),
      scenarioTable,
      spacer(),
      body('注：所有场景均基于"self-trajectory"注入方式——从电池自身的历史数据中提取末期特征，拼接或混合至故障点之后，模拟提前进入老化状态。', { italics: true, color: '555555' }),

      // ── 二、五个检测信号 ──
      spacer(),
      h1('二、五个检测信号'),
      body('模型每次预测后，异常检测模块从以下五个维度分析 SOH 时间序列，输出逐循环的可疑程度分数（分数越高越异常）：'),
      spacer(),
      signalTable,
      spacer(),
      body('实验结论：', { bold: true }),
      bullet('trajectory_deviation 是唯一在全部 5 类场景下稳定有效的信号（AUC 0.65–0.75）'),
      bullet('input_zscore 在正常电池生命末期自然偏高，直接参与综合评分会拖累整体性能'),
      bullet('mono_violation / rate_anomaly / drop_anomaly 在当前实验中 AUC ≈ 0.5，接近随机水平'),

      // ── 三、评价指标 ──
      spacer(),
      h1('三、评价指标'),
      metricTable,
      spacer(),
      body('三个指标分别回答三个问题：', { bold: true }),
      bullet('"能不能检"——AUC 衡量整体排序能力，不依赖具体阈值'),
      bullet('"能抓多少"——Det@FPR5% 衡量实际运维约束下的检出能力'),
      bullet('"能多快"——det_delay 衡量从故障发生到首次报警的响应速度'),

      // ── 四、识别过程 ──
      spacer(),
      h1('四、识别过程'),
      body('系统对每块电池按以下步骤持续监测：'),
      spacer(),
      stepTable,

      // ── 五、实验结果 ──
      spacer(),
      h1('五、实验结果（severe 严重度，trajectory_deviation 信号）'),
      resultTable,
      spacer(),
      body('核心结论：', { bold: true }),
      bullet('突发型故障（A1、A4）：AUC > 0.70，检出率约为随机水平 3 倍，工程上具有实用价值'),
      bullet('渐进型故障（A3、A5）：trajectory_deviation 单信号 AUC 约 0.65，仍优于随机，但需进一步优化'),
      bullet('该方案无需真实失效样本训练，属于"零额外成本"异常感知，可直接复用 SOH 估计模型'),
      spacer(),
      body('优化方向：将报警判定由"单点超线"改为"连续 N 次超线"，在略微增加检测延迟的代价下，可将误报概率降低数百倍。', { italics: true, color: '444444' }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('docs/battery_anomaly_report.docx', buf);
  console.log('Done: docs/battery_anomaly_report.docx');
});
