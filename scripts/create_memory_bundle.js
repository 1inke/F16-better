const fs = require('fs');
const path = require('path');
const { serializeBundle } = require('C:/Users/linke/.dsh/profiles/web/node_modules/dsh-butler/lib/types/portability/codec.js');

const outDir = path.resolve(__dirname, '..', 'offline-memory-bundles');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

const entries = [
  {
    exportEntryId: 'srs-001',
    key: 'research.baseline_identity_lock',
    type: 'constraint',
    originalScope: 'global',
    category: 'scientific_methodology',
    suggestedExpiration: null,
    sourceMetadata: {
      originType: 'synthesis',
      sanitizedSummary: 'Baseline Identity严格锁定规则'
    },
    content: '【Baseline Identity 严格锁定】：任何比较只有在架构、输入维度、特征组、归一化、损失函数、优化器、超参数（epochs/lr/batch）、随机种子策略、内外折划分、校准流程与权重哈希完全一致时，方可视为同一 baseline；任何一项不一致禁止横向比较归因；禁止使用硬编码的历史数值假充当前 baseline 运行结果。'
  },
  {
    exportEntryId: 'srs-002',
    key: 'research.no_hardcoded_metrics',
    type: 'constraint',
    originalScope: 'global',
    category: 'scientific_methodology',
    suggestedExpiration: null,
    sourceMetadata: {
      originType: 'synthesis',
      sanitizedSummary: '单一数据源与禁止硬编码实验结果'
    },
    content: '【单一数据源与禁止硬编码实验指标】：Markdown、README、分析总结与脚本中严禁手工填写或硬编码指标数值（ADE/FDE/碰撞率/Pairwise/参数量/FLOPs/延迟等）；所有结果必须从机读输出（JSON/CSV）中动态加载渲染；模型参数量与维度必须从真实模型张量（如 x.shape[-1]）推导计算，发现报告与机读结果不一致必须立即停机报警。'
  },
  {
    exportEntryId: 'srs-003',
    key: 'research.claim_strength_levels',
    type: 'preference',
    originalScope: 'global',
    category: 'scientific_methodology',
    suggestedExpiration: null,
    sourceMetadata: {
      originType: 'synthesis',
      sanitizedSummary: '结论强度分级与证据优先原则'
    },
    content: '【结论强度分级与证据优先原则】：分析严格遵循 Observation -> Validity Check -> Alternative Explanations -> Conclusion，禁止跳跃式定论；默认采用数据充分支持的最弱表述；探针未检出信号时仅表述为“在当前测试表征、容量和协议下未发现可恢复信号”，严禁在无严格理论检验下升级为“information-theoretic limit / 证明表示极限 / 彻底证实”等过度表述。'
  },
  {
    exportEntryId: 'srs-004',
    key: 'research.data_metric_discipline',
    type: 'constraint',
    originalScope: 'global',
    category: 'scientific_methodology',
    suggestedExpiration: null,
    sourceMetadata: {
      originType: 'synthesis',
      sanitizedSummary: '数据索引显式对齐与指标单元测试'
    },
    content: '【数据索引对齐、指标测试与宏微观隔离】：样本筛选、随机抽样与缓存必须保留显式 sample_id / index，严禁隐式切片对齐；新增/修改指标必须先通过极简人工 toy 测试验证正反向边界；严守测试集隔离，禁止 transductive scaling 与测试集超参选择；严格区分 Macro 与 Micro，多折数据以官方 Macro 为核心准则，严禁以被大折主导的 Micro 优势替代 Macro 结论。'
  },
  {
    exportEntryId: 'srs-005',
    key: 'research.experiment_scope_discipline',
    type: 'constraint',
    originalScope: 'global',
    category: 'scientific_methodology',
    suggestedExpiration: null,
    sourceMetadata: {
      originType: 'synthesis',
      sanitizedSummary: '实验范围纪律与数学变换自检'
    },
    content: '【实验范围纪律与数学变换自检】：严格执行方案设定的阶段决策暂停点，未获人工确认前严禁擅自级联执行后续实验；进行中心化、归一化、投影等数学变换后，必须先进行数学性质自检（Sanity Check），明确区分零和约束等定义导致的数值伪象与真实数据结构变化，严禁将数学必然伪象赋予业务含义。'
  }
];

const header = {
  exportId: 'bundle-scientific-rigor-standards-v1',
  createdAt: '2026-09-21T15:00:00.000Z',
  toolVersion: '1.0.0',
  description: 'AI实验研究执行规范与科研工程纪律准则（Baseline Identity, No Hard-coded, Claim Strength）'
};

const bundleJson = serializeBundle(header, entries);
const bundlePath = path.join(outDir, 'scientific-rigor-standards.json');
fs.writeFileSync(bundlePath, bundleJson, 'utf8');
console.log('Successfully generated:', bundlePath);
