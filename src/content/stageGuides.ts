export interface GuideStep {
  /**
   * Primary glossary key for this step. Its term + long description head the
   * step. Optional: extra/overview steps (asides, analogies) can omit it and
   * supply a `title` instead.
   */
  glossaryKey?: string;
  /** Heading for steps without a glossaryKey (e.g. overviews, analogies). */
  title?: string;
  /**
   * Step body. May contain inline glossary markup: [[key]] renders the term's
   * display name as a clickable popup, [[key|label]] uses a custom label.
   */
  note?: string;
}

export interface StageGuide {
  stageId: number;
  title: string;
  steps: GuideStep[];
}

export const stageGuides: Record<number, StageGuide> = {
  1: {
    stageId: 1,
    title: "Single-Request Lifecycle",
    steps: [
      {
        title: "Overview",
        note: "A single request moves through two phases. [[prefill|Prefill]] processes all prompt tokens in one pass to populate the [[kvCache|KV cache]]. Then [[decode]] generates one new token per step, each step [[vocabSampling|sampling]] from the full vocabulary, until the limit is reached and the request finishes. The steps below walk through each part.",
      },
      {
        glossaryKey: "tokenizer",
        note: "Type a sentence in the prompt box. The tokenizer splits your text into subword tokens — each word may become one or more token ids that the model actually processes.",
      },
      {
        glossaryKey: "prefill",
        note: "Watch all prompt tokens get processed in a single forward pass. The KV cache fills up instantly for every input token before the model produces even one output token.",
      },
      {
        glossaryKey: "decode",
        note: "Now the model generates output one token at a time. Each step reads the full KV cache and appends one new token — you can see the output grow word by word.",
      },
      {
        glossaryKey: "vocabSampling",
        note: "At every decode step the model scores all vocabulary entries and samples the next token. How that choice is made — temperature, top-k, top-p — is a model concept explored on the next stage.",
      },
      {
        glossaryKey: "kvCache",
        note: "Notice the KV cache bar grow with each new token. This memory is never freed until the request completes — it is the main cost of serving a long conversation.",
      },
    ],
  },

  2: {
    stageId: 2,
    title: "Model Concepts",
    steps: [
      {
        title: "Overview",
        note: "A quick aside on the model itself. The serving stages treat the model as a black box that emits one token per [[decode]] step; this stage peeks at how that token is chosen. These are model/decoding properties, independent of how vLLM batches or schedules requests.",
      },
      {
        glossaryKey: "vocabSampling",
        note: "Each step produces a probability over the whole vocabulary. A sampling strategy turns that distribution into one chosen token. Drag the knobs in the panel to reshape it.",
      },
      {
        glossaryKey: "temperature",
        note: "[[temperature|Temperature]] scales the logits before softmax. Below 1 sharpens toward the top token (focused, more deterministic); above 1 flattens for variety. At 0 it becomes [[greedy|greedy]] — always the single most-likely token.",
      },
      {
        glossaryKey: "topP",
        note: "[[topK|Top-k]] keeps the k most-likely tokens; [[topP|top-p]] keeps the smallest set whose probabilities sum to p. Both truncate the unlikely tail before sampling — struck-through tokens in the panel are the ones excluded.",
      },
    ],
  },

  3: {
    stageId: 3,
    title: "Paged KV Cache",
    steps: [
      {
        title: "Overview",
        note: "The [[kvCache|KV cache]] is split into fixed-size physical [[block|blocks]] (e.g. 4 tokens each). Instead of one huge contiguous slab per request, each request gets a [[blockTable|block table]] — a logical→physical map like an OS page table. This is [[pagedAttention|paged attention]]: a sequence's KV data need not be contiguous in memory. As a request grows, new blocks are allocated on demand; when it finishes, all its blocks are freed instantly. Only the last partial block introduces [[internalFragmentation|internal fragmentation]] — bounded to <1 block per request.",
      },
      {
        glossaryKey: "kvCache",
        note: "With multiple requests in flight the KV cache becomes the bottleneck. Naively, each request would need a contiguous buffer reserved for its maximum possible length — most of it unused.",
      },
      {
        glossaryKey: "block",
        note: "vLLM carves the KV cache into small fixed-size blocks. Each block holds the K/V tensors for a handful of tokens. New blocks are allocated on demand as the sequence grows.",
      },
      {
        glossaryKey: "pagedAttention",
        note: "With paged attention the model never needs a contiguous max-length reservation. Any free block can be assigned to any request, so short and long requests coexist without wasting memory.",
      },
      {
        glossaryKey: "blockTable",
        note: "Each request maintains its own block table mapping logical positions to physical block ids. The attention kernel uses this table to assemble K/V data scattered across non-contiguous blocks.",
      },
      {
        glossaryKey: "internalFragmentation",
        note: "The only wasted space is in the last block of each sequence — at most (block_size - 1) token slots. Compare this to the potentially thousands of wasted slots in contiguous allocation.",
      },
    ],
  },

  4: {
    stageId: 4,
    title: "Continuous Batching",
    steps: [
      {
        title: "Overview",
        note: "Many users hit the server at once. [[continuousBatching|Continuous batching]] rebuilds the running batch every step: requests wait in the [[waitingQueue|waiting queue]], get admitted to the [[runningQueue|running queue]] up to [[maxBatchSize|Max Batch Size]] and the [[tokenBudget|token budget]], and are retired the instant they finish. When the cache fills, [[preemption]] evicts a running request ([[swapping|swapped]] back to the queue). The steps below let you drive each behavior.",
      },
      {
        glossaryKey: "waitingQueue",
        note: "Simulate multiple users by submitting several prompts. Each new request lands in the waiting queue and waits for the scheduler to admit it.",
      },
      {
        glossaryKey: "runningQueue",
        note: "The running queue shows who is active right now. With static batching this would be fixed until everyone finishes; with continuous batching it changes every step.",
      },
      {
        glossaryKey: "continuousBatching",
        note: "As soon as one request completes, its slot is freed and the next waiting request is admitted — no wasted GPU cycles waiting for slower requests to catch up.",
      },
      {
        glossaryKey: "maxBatchSize",
        note: "Raise [[maxBatchSize|Max Batch Size]] (under Scheduler) for more concurrent requests and higher throughput — until KV cache pressure and memory bandwidth become the bottleneck. Real servers run 256+; 1 just serializes requests.",
      },
      {
        glossaryKey: "tokenBudget",
        note: "The [[tokenBudget|token budget]] caps total tokens admitted per step, so one expensive prefill can't block the existing batch's decode steps. Add several requests at once and watch some defer to a later step.",
      },
      {
        glossaryKey: "chunkedPrefill",
        note: "Production schedulers go further with [[chunkedPrefill|chunked prefill]]: a long prompt's prefill is sliced into token-budget-sized chunks and interleaved with ongoing decodes, so it never monopolizes a step. (This demo prefills in one tick for simplicity — chunked prefill is the real refinement of that step, and a big reason colocated serving competes with PD disaggregation.)",
      },
      {
        glossaryKey: "preemption",
        note: "Set a small [[kvCache|KV Cache Blocks]] under Engine Setup, then flood the scheduler with requests. When blocks run out, [[preemption]] evicts a running request — watch it leave the running queue.",
      },
      {
        glossaryKey: "swapping",
        note: "The preempted request becomes [[swapping|swapped]] and returns to the queue. It is re-admitted and re-prefilled when completions free enough blocks — demonstrating the cost of cache pressure.",
      },
      {
        title: "If you've trained a model: where the token budget comes from",
        note: "In training you push a batch of batch_size × seq_len tokens through one forward pass — uniform length, all tokens known. Serving is the same pass but ragged and rebuilt each step. seq_len → a request's prompt length (a [[prefill]] step processes the whole prompt at once). batch_size → requests packed together, capped by [[maxBatchSize|Max Batch Size]]. The new part is [[decode]]: each decoding request adds just 1 token per step while a prefilling one adds its full prompt. The [[tokenBudget|token budget]] is the cap on total tokens per pass — your batch_size × seq_len, now summed over a mix of big prefills and 1-token decodes.",
      },
      {
        title: "Try it",
        note: "Flood the scheduler with the Load Generator or repeated Add. Set a small [[kvCache|KV Cache Blocks]] first to trigger [[preemption]] under load, and select a running request chip then Cancel to see its blocks reclaimed instantly.",
      },
    ],
  },

  5: {
    stageId: 5,
    title: "Parallelism Strategies",
    steps: [
      {
        title: "Overview",
        note: "Large models are spread across many [[gpu|GPUs]], and the four strategies split different things: [[tensorParallel|tensor parallelism]] splits weight matrices within each layer, [[pipelineParallel|pipeline parallelism]] splits layers across GPUs, [[expertParallel|expert parallelism]] splits MoE experts, and [[dataParallel|data parallelism]] splits the request batch. Toggle the modes to see how the same GPUs are used differently.",
      },
      {
        glossaryKey: "tensorParallel",
        note: "Tensor parallelism shards each layer's weight matrix across GPUs. Toggle to TP and observe how every GPU holds a slice of every layer — all-reduce is needed after each layer to combine results.",
      },
      {
        glossaryKey: "pipelineParallel",
        note: "Pipeline parallelism assigns different layers to different GPUs. Switch to PP and see how GPU 0 runs early layers, GPU 1 middle layers, etc. — requests flow through GPUs in sequence.",
      },
      {
        glossaryKey: "expertParallel",
        note: "Expert parallelism distributes MoE expert blocks across GPUs. Each token is routed to the GPU hosting its chosen experts — great for MoE models like Mixtral and DeepSeek.",
      },
      {
        glossaryKey: "dataParallel",
        note: "Data parallelism replicates the full model on every GPU and divides the request batch. Toggle to DP — each GPU runs independently with no cross-GPU communication during inference.",
      },
      {
        glossaryKey: "gpu",
        note: "Each box represents one GPU. Notice how the same N GPUs are used differently depending on the parallelism strategy — splitting weights, layers, experts, or requests.",
      },
      {
        title: "In practice",
        note: "Each strategy has a different communication cost and suits a different scenario, so real deployments combine them — e.g. [[tensorParallel|TP]] within a node (fast interconnect) and [[dataParallel|DP]] across nodes.",
      },
    ],
  },

  6: {
    stageId: 6,
    title: "Model Features: FP8 & MoE",
    steps: [
      {
        title: "Overview",
        note: "Two orthogonal efficiency techniques. [[fp8|FP8]] [[quantization]] stores weights and KV in 1 byte instead of 2, halving memory with little accuracy loss. [[moe|Mixture of Experts]] adds many expert FFN blocks but activates only top-k per token, so [[activeParams|active parameters]] (compute) stay small while total params (capacity) grow. Toggle each below; they combine in models like DeepSeek.",
      },
      {
        glossaryKey: "quantization",
        note: "Quantization reduces the numerical precision of weights and activations. Toggle between FP16 and FP8 to see how the memory bars change — lower precision means smaller storage footprint.",
      },
      {
        glossaryKey: "fp8",
        note: "FP8 uses 1 byte per element vs FP16's 2 bytes. Watch the weight and KV cache bars halve when you switch to FP8. Modern GPUs have native FP8 hardware, so throughput improves too.",
      },
      {
        glossaryKey: "moe",
        note: "Toggle MoE on to see the Mixture-of-Experts FFN layer. A router selects top-k of E experts per token — the unselected experts (grayed out) are completely skipped during inference.",
      },
      {
        glossaryKey: "expertRouting",
        note: "Each token's routing is determined by learned gating weights. The router assigns tokens to the experts they are most likely to benefit from — different tokens often land on different experts.",
      },
      {
        glossaryKey: "activeParams",
        note: "The key insight: total parameters (memory cost) can be huge, but active parameters (compute cost per token) is only top-k experts worth. MoE scales capacity without scaling per-token FLOPs.",
      },
    ],
  },

  7: {
    stageId: 7,
    title: "Attention Techniques",
    steps: [
      {
        title: "Overview",
        note: "Attention is where tokens talk to each other — and where memory costs accumulate. This stage goes from fundamentals to optimizations: [[scaledDotProduct|how attention is computed]] (Q/K/V), [[flashAttention|how it's made IO-efficient]] (Flash Attention), and two families of tricks that shrink the [[kvCache|KV cache]]: [[slidingWindow|bounding lookback]] and [[attentionVariant|reducing KV heads]].",
      },
      {
        glossaryKey: "scaledDotProduct",
        note: "Step through the Q/K/V panel. Each token is projected into [[queryMatrix|Q]], [[keyMatrix|K]], and [[valueMatrix|V]]. Q × Kᵀ produces raw [[attentionScore|attention scores]], [[softmax]] turns them into weights, and those weights blend the V rows into the output.",
      },
      {
        glossaryKey: "flashAttention",
        note: "Standard attention writes the full N×N score matrix to [[hbm|HBM]] — expensive at long sequences. [[flashAttention|Flash Attention]] tiles the computation so only one block at a time lives in fast [[sram|SRAM]], cutting HBM reads from O(N²) to O(N). Toggle Standard vs Flash to see the difference.",
      },
      {
        glossaryKey: "slidingWindow",
        note: "[[slidingWindow|Sliding-window]] attention limits each token to looking back at most W positions, so the KV cache stays O(W) instead of O(length). Drag the W slider: only the last W tokens stay live; older ones are evicted. This is what makes 100k–1M-token contexts affordable.",
      },
      {
        glossaryKey: "attentionVariant",
        note: "The number of stored K/V heads sets the cache size. [[mha|MHA]] keeps one per query head (biggest); [[gqa|GQA]] shares them in groups (~4× smaller, the modern default); [[mla|MLA]] compresses K/V to one latent per token (smallest, but needs DP-attention under TP — see the SGLang stage).",
      },
    ],
  },

  8: {
    stageId: 8,
    title: "Speculative Decoding",
    steps: [
      {
        title: "Overview",
        note: "[[decode|Decode]] is memory-bandwidth-bound — each token re-reads the whole model for one position. [[speculativeDecoding|Speculative decoding]] reads it once for several positions: a small draft guesses ahead, the target verifies in parallel, so one expensive target pass yields several tokens — with provably identical output.",
      },
      {
        glossaryKey: "draftModel",
        note: "A small, fast [[draftModel|draft model]] proposes the next k tokens. Its guesses needn't be perfect — wrong ones are caught — but the closer it is to the target, the higher the acceptance and the bigger the win. Use the k slider to set how far it guesses ahead.",
      },
      {
        glossaryKey: "verification",
        note: "The target [[verification|verifies]] all k drafted tokens in one forward pass (same cost as decoding one token), accepts the longest correct prefix, and corrects the first mismatch. Green = accepted, red = first reject, the rest are discarded.",
      },
      {
        glossaryKey: "acceptanceRate",
        note: "The [[acceptanceRate|acceptance rate]] sets the realized speedup: each pass yields ~1 + accepted tokens instead of 1. Push acceptance and k up to watch the speedup grow; a low acceptance rate wastes the draft's guesses.",
      },
    ],
  },
  9: {
    stageId: 9,
    title: "PD Disaggregation",
    steps: [
      {
        title: "Overview",
        note: "[[pdDisaggregation|PD disaggregation]] runs the two phases on separate GPUs. The [[prefillWorker|prefill worker]] is compute-bound (whole prompt in one bursty pass); the [[decodeWorker|decode worker]] is memory-bandwidth-bound (one token per step over a large KV cache). When prefill finishes, the KV is shipped across the interconnect in a [[kvTransfer|KV transfer]] — letting each worker type be sized and optimized independently.",
      },
      {
        glossaryKey: "pdDisaggregation",
        note: "PD disaggregation splits prefill and decode work onto separate GPU workers. Add a request and watch it appear on the Prefill Worker panel first.",
      },
      {
        glossaryKey: "prefillWorker",
        note: "The prefill worker processes all prompt tokens in a single compute-intensive burst. It is optimized for high throughput on large token batches — very different from steady decode work.",
      },
      {
        glossaryKey: "kvTransfer",
        note: "When prefill completes, the computed KV cache is shipped to the decode worker. Watch the transfer indicator light up at the prefill→decode boundary.",
      },
      {
        glossaryKey: "decodeWorker",
        note: "The decode worker takes over and generates tokens one per step, attending over the transferred KV cache. It runs independently from new prefills happening on the prefill worker.",
      },
      {
        glossaryKey: "weightReplication",
        note: "A subtlety: the pools are separate GPUs, so each holds its own full copy of the weights — [[weightReplication|weight replication]]. This is NOT 2× the hardware, though: the same 8 GPUs that run one TP-8 instance can be split into prefill-TP4 + decode-TP4 (two copies, each sharded over 4 GPUs) — same GPUs, same cost. The real limit is per-pool: each pool must collectively fit a full copy, so a model that nearly fills your whole budget as one instance can't be partitioned on the same hardware and must scale out.",
      },
      {
        title: "Is it worth it?",
        note: "For a small deployment, usually not — just run a couple of colocated TP-8 replicas behind a load balancer: traffic stays node-local (no cross-node [[kvTransfer|KV transfer]]), replicas are fungible (either serves any request, no fixed prefill:decode ratio to mis-guess), and they fail uniformly. [[continuousBatching|Chunked prefill]] already interleaves prefills with decodes, weakening the original 'prefill stalls decode' motivation. Disaggregation pays off at large scale — when you're already running many model replicas and can right-size dedicated prefill and decode pools (different parallelism, ratios, even hardware) to hit tight TTFT+ITL SLOs on predictable, skewed workloads. There the gains from phase isolation outweigh the transfer cost and ratio-mismatch risk.",
      },
    ],
  },

  10: {
    stageId: 10,
    title: "SGLang vs vLLM",
    steps: [
      {
        title: "Same fundamentals, narrowing gap",
        note: "Everything in Stages 1–6 applies to both engines. As both matured the feature gap narrowed: today both support automatic [[prefixCache|prefix caching]], tensor/[[expertParallel|expert]]/data parallelism, and [[dpAttention|DP-attention]]. The remaining differences are mostly data structures, defaults, and which optimizations each pioneered — not 'one can, the other can't.' This stage looks at two instructive contrasts.",
      },
      {
        glossaryKey: "prefixCache",
        note: "Stages 1–6 are common to both engines. Both support automatic prefix caching: when requests share a prompt prefix — a system prompt, few-shot examples, chat history — the KV for that prefix is reused instead of recomputed. SGLang enables this by default; vLLM requires enable_prefix_caching=True (on by default in the V1 engine). Toggle the shared-prefix requests to see how much overlaps.",
      },
      {
        glossaryKey: "radixAttention",
        note: "The difference is the data structure, not the capability. vLLM hashes blocks of tokens and matches them in a flat pool; SGLang structures the whole cache as a radix tree that branches at the first differing token. Both reuse the shared prefix — the tree just makes overlapping sub-branches (multi-turn, tree-of-thought) especially natural.",
      },
      {
        glossaryKey: "lruEviction",
        note: "Both engines evict least-recently-used entries when memory is tight (a free queue in vLLM, leaf eviction in SGLang's tree), keeping hot shared prefixes resident.",
      },
      {
        glossaryKey: "dpAttention",
        note: "Where KV lives and where experts live are independent axes. Plain tensor-parallel attention shards KV across ranks — except for DeepSeek-style MLA, whose tiny latent KV ends up replicated. DP-attention fixes that by giving each rank distinct requests' KV. Pioneered in SGLang, now in vLLM too.",
      },
      {
        glossaryKey: "expertParallel",
        note: "Expert parallelism is a separate axis from KV placement: it shards the MoE FFN's experts across GPUs, routing each token (all-to-all) to whichever rank holds its top-k experts. It's about weights/compute, not KV — and both engines support it (with DeepEP backends and load balancing). It is not SGLang-only.",
      },
      {
        glossaryKey: "programmableFrontend",
        note: "SGLang also exposes a programmable frontend: you script multi-step LLM programs and the runtime shares KV across branches and batches parallel calls automatically — it sees structure that opaque requests hide.",
      },
      {
        glossaryKey: "overlappedScheduling",
        note: "SGLang overlaps CPU scheduling and GPU compute (and expert-parallel communication) so the GPU rarely stalls. The honest takeaway: both engines share the building blocks today — the differences are defaults, maturity, and which optimizations each pioneered, not raw capability.",
      },
    ],
  },
};
