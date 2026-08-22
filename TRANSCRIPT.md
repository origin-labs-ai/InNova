Context Ye to bas halka sa part dekha hai tumne iska, README poora padho tab samjhoge! Usme dekhna Kimi K3 and DeepSeek V4 Flash se related kitni plannings hai and kitne features add karne ka socha gaya hai project me! Aur "Autograd / Training Engine: Autograd DAG, AdamW, Straight-Through Estimator (STE) aur native fine-tuning support." Bhai maine to yahaa Adafactor daalne bola tha to AdamW kaise aaya?! Fix karna hai ye bhi! Aur poore llama.cpp me jitne backends hai jitne type ke sab daalna hai saath hi cuda bhi daalna hai! But without dependencies! Poore project me saare ke saare features apply karne hai, saare formats ka test karna hai, saare QUANT varients ko Q varients banaana hai jaise QUANT1 banega Q1 and QUANT2 banega Q2 and yahi saare varients pe apply hoga! Aur Maine suna hai ki ek baar AI model is tarah se train ho gaya to wo us continual loop me fas jaayega ki baad me kharaab ho jaayega ye aisa AGI engine hai so isko bhi fix karna hai! Koi jugaar nikaal ke! Uske baad sun, sirf Q1, Q2, Q3, Q4, Q6, Q8, Q12, Q16, Q24, Q32 hi rakhna hai and Q16 ka codec thora better karke FP16 ko beat karna hai nahi to sab bolenge ki FP16 direct copy maar diya hu mai, saare industrial varients ko haraana hai unke is competetor se and nakli me nahi, asli me and real weights pe test honge model download karke and guassian and ek head-to-head hoga and sabka chart banayega tum ki ham hi better hai har jagah! Uske baad ka rule ye hai ki BPW se compromise mat kar lena yaani BPW mat badhaa dena jo jitna BPW waala varient hai usko utna hi milega, GRP grouping waalo ko x2 ko haraana parega jaise ki Q4_GRP waale ko Q8 and baaki ke 8 BPW varients ko haraana parega and yahi rule sabpe hai jaise ki 4x2=8 and aise hi saare GRP varients me hai while MIXED waalo ko to FP32 level quality deni hi hai, FP32 me sabko baraabar priority milti hai kharaab waale weights ko bhi jinko nahi deni hai lekin unko mixed ke through better banaake kar sakte hai, ham hi best banenge! Ye sab kuchh karna hai and saare varients ke GRP honge existing varients hataake saare naye BPW waale varients daal do exact same BPW pe and sun lo, mixed me 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5, 16.5, 24.5 tak hi rahenge exact and 2 TWI_MIX me jaayenge baaki ke QUAD_MIX me aur mixed ke bhi GRP varients and GRP me real grouping jaise use kari gayi hai waise hi tumko bhi karni hai! Sab kuchh karna hai! Tab Comparison table and charts/visuals banaane hai, project ko production ready banaana hai! Sab kuchh karo! Lag jaao! Saath me "Bhai, tune ekdum kadak cheez pakdi hai! Ye thecodacus/llama.cpp ka fork (specifically fable5/prefetch-experts branch) MoE (Mixture of Experts) models ke liye ekdum game-changer hai . Asli speed boost 65% nahi, balki 64% (+64%) hai, jo ki local AI inference ke hisaab se massive hai 

github.com

. Main tujhe iski har ek detail, code logic aur apne fork me kaise laye, wo ultra-detailed way me samjhata hoon.

1. Asli Problem Kya Thi? (Why was it slow earlier?)

Jab tu MoE models (jaise Qwen3.6-35B ya Mixtral) ke experts ko CPU RAM me offload karta hai (--n-cpu-moe flag use karke), toh prefill (prompt processing) ke time GPU ekdum idle baitha rehta tha 

github.com

.

Serial Execution: Mainline llama.cpp me pehle Layer N ke weights CPU se GPU (H2D copy) upload hote the, fir Layer N compute hota tha, aur fir Layer N+1 ke weights upload hote the 

github.com

. Ye sab serial me hota tha, isliye GPU idle rehta tha 

github.com

.

The Sync Point (D2H Readback): Ye sabse badi dikkat thi. GPU ko pata hona chahiye ki kaunse experts upload karne hain 

github.com

. Toh har layer ke baad GPU routing IDs ko wapas CPU (D2H readback) bhejta tha 

github.com

. Ye ek "hard sync point" ban jata tha 

github.com

. Pehle compute khatam, fir IDs aaye, fir upload shuru 

github.com

. Is sync ki wajah se overlap karna impossible tha 

github.com

.

2. 64% Speed Kaise Badhi? (The 2 Major Optimizations)

Is fork me 2 main optimizations hain jo environment variables se enable hote hain aur output ekdum token-identical rehta hai .

Optimization A: Page-Locking CPU Memory (GGML_CUDA_REGISTER_HOST=1) 

Kya hota tha: Pehle CPU RAM me mmap'd expert weights pageable memory me hote the . Jab GPU unhe PCIe ke through copy karta tha, toh wo driver ke ek hidden "bounce buffer" me jate the . Iski speed sirf 6-7 GB/s hoti thi .

Kya kiya: Is fork ne CPU memory ko "page-lock" (pin) kar diya . Isse H2D (Host-to-Device) copies direct DMA (Direct Memory Access) ke through PCIe par hone lage .

Result: Transfer speed seedha ~20 GB/s ho gayi!  Is akele change se ~21% speed boost mil gaya 

github.com

.

Optimization B: Asynchronous Prefetching & Overlapping (GGML_SCHED_PREFETCH_EXPERTS=1) 

Kya kiya: Ek second CUDA stream banaya gaya . Ab jab Layer N GPU par compute ho raha hota hai, tab theek usi time second stream Layer N+1 ke weights ko CPU se GPU par upload (prefetch) kar raha hota hai . Compute aur upload parallel/overlap ho gaye .

The Masterstroke (Sync Bypass): Bade batch sizes (jaise 2048 tokens prefill) me router waise bhi lagbhag saare experts (e.g., 256/256) ko select kar leta hai 

github.com

. Is fork ne ek threshold set kiya—agar batch size bada hai, toh wo D2H readback (routing IDs CPU ko bhejna) ko skip kar deta hai 

github.com

. Seedha saare experts upload kar diye jate hain 

github.com

. Sync point hatne ki wajah se second stream bina ruke aage badh jata hai 

github.com

.

Result: Prefill ke dauran GPU ka idle time ~42% se ghat kar sirf "few percent" reh gaya 

github.com

.

3. Benchmark Details (Proof)

Hardware: RTX 3060 12GB 

Model: Qwen3.6-35B-A3B (--n-cpu-moe 26) 

Task: 2048 prompt tokens processing (prefill) 

Before: ~1143 tokens/sec 

After: ~1880 tokens/sec (+64%) 

4. Apne Fork Me Ye Feature Kaise Daalein? (Implementation Guide)

Agar tera khud ka llama.cpp fork hai aur tu ye directly code me merge karna chahta hai, toh Git aur Code level par ye karna padega:

Tarika 1: Git Merge (Sabse Aasan)

bash

# 1. Apne fork/clone me jaocd your-llama-cpp-fork# 2. thecodacus ko remote add karogit remote add thecodacus https://github.com/thecodacus/llama.cpp.gitgit fetch thecodacus# 3. Unki branch ko merge ya cherry-pick karogit merge thecodacus/fable5/prefetch-experts

Tarika 2: Code/Logic Level (Agar manually implement karna hai)

Agar tu khud C++ code likhna chahta hai, toh ye 4 steps follow kar:

Memory Pinning: Jahan GGUF file mmap hoti hai (ggml-backend-reg.cpp ya CUDA backend me), wahan expert weights ke pointers par cudaHostRegister(ptr, size, cudaHostRegisterDefault) call karna padega jab GGML_CUDA_REGISTER_HOST=1 set ho . Isse memory pin ho jayegi.

CUDA Streams: CUDA backend me ek naya stream initialize karna hoga (e.g., cudaStream_t copy_stream).

Async Overlap: llama.cpp ke compute graph me, H2D copies ko cudaMemcpyAsync(..., copy_stream) par daalna hoga.

Condition for Prefetch: llama_decode ya graph execution me ek condition daalni hogi: if (batch_size > threshold && is_prefill). Agar true hai, toh router ke D2H sync ko bypass karna hai aur directly cudaMemcpyAsync se saare experts copy karne hain, taaki compute aur copy overlap ho sake bina pipeline ruke.

5. Ise Run Kaise Karein?

Jab tu ise compile kar lega, toh bas terminal me ye environment variables set karke llama-bench ya llama-cli run karna:

bash

GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 ./build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub Bhai, ye MoE offload ke liye absolute goldmine hai. Agar tujhe C++ code me exact file changes dekhni hain (jaise ggml-cuda.cu me kahan kya badla), toh bata, main wo bhi nikaal dunga! 🚀" ye feature bhi daalna hai, apne tareeke se! Existing BPW waalo ko delete karna but tareeka dekh lena sabka! Exact jitne BPW mai bataaya hu bas wahi honge and sabke GRP honge and mixed me mai jitne bola bas wahi honge! Quality giri to tu mara samajh le! Lag jaa! Har cheej karni hai!


context Abe "Q16(30%) + Q24(50%) + Q32(20%)" ye kaisa QUAD_MIX k=hai be?! Fix kar isko implementation plan me! Sab kuchh theek se karna, quality best hi chahiye, cheating bhi mat karna, stubs mat likhna and koi issue/problem/warnings/bugs ya kuchh mat chhorna and production ready bhi banaana, har baar 5 sub-agents bhej ke poora codebase ka audit karwaana and saare problems fix karna saare implementations ke then firse bhejna agents and jis baar 5 ke 5 se exact infinite/10 yaani exact infinite/10 level quality hai aisa mil jaaye jisme wo jhooth naa bol rahe ho tum khud check kar lena tabhi rukna! Aur maine bola tha README me Kimi K3 and DeepSeek V4 Flash aur bahut saare plans hai and 256+ features hai, sab complete karne hai be! Ia am serious! Planning complete kar then implement complete karke hi rukna! Jab production ready ho jaaye kahi koi stubs naa mile poore project me and har feature poori tareeke se kaam karne lage bestest way me tabhi rukna! Bhai firse check karle saare conditions match hone chahiye, tujhe khud pata hai ki ye kitna powerful and ambitious project hai saath me kitna serious hai! Lag jaa! Plan complete karke uske according implement karne lag jaana!

context Abe ye kya hai be types.h me "QUANT1          = Q1,
    QUANT2          = Q2,
    QUANT4          = Q4,
    QUANT8          = Q8,
    QUANT16         = Q16,
    QUANT32         = Q32,
    QUANT1_GRP      = Q1_GRP,
    QUANT2_GRP      = Q2_GRP,
    QUANT4_GRP      = Q4_GRP,
    QUANT8_GRP      = Q8_GRP,
    QUANT16_GRP     = Q16_GRP,
    QUANT_Q0        = Q_TWI_MIX_1_5,     // Legacy: 1.5 BPW → maps to TWI_MIX@1.5
    QUANT_Q0_GRP    = Q_TWI_MIX_1_5_GRP, // Legacy: 1.5 BPW GRP → maps to TWI_MIX@1.5_GRP
    QUANT_Q1        = Q2,                 // Legacy: 2.0 BPW sparse → maps to Q2
    QUANT_Q1_GRP    = Q2_GRP,             // Legacy: 2.0 BPW sparse GRP → maps to Q2_GRP
    QUANT_6_K       = Q6_GRP,             // Legacy: 6.5625 BPW → maps to Q6_GRP" dekh! Saala QUANT hataane bola tha mai! Baaki ke kaam bhi complete kar! Goal pe lage reh!

    context # Gauntlet Loop

Disciplined loop for top-tier work in any domain: **split → build → blind-critic → repeat**, against a hard **bar** the agent cannot talk its way past. The technique is Matt Shumer's (Claude of Duty); this merged skill combines the best of 7 community variants.

The loop produces quality only because the thing it compares against is real. Everything else is scaffolding.

## Flow

1. **Read the goal.** One-line restatement in your head, not on screen.
2. **Set the bar.** If the user supplied a reference, use it. If not, offer **2 or 3 candidate bars**, one line each, and stop. Wait for their pick. Do not start building yet.
3. **Run the loop** (below). For a simple goal, the loop is: split → build → blind-critic → fix → repeat until the critic picks ours.
4. **Report.** Final artifact + the bar used + a round log + PASS evidence + anything still under the bar.

## The bar is the whole trick

A bar must pass three tests:

- **Named.** A specific thing, not a category. "Stripe's pricing page" works. "Award-winning SaaS sites" does not.
- **Fetchable.** The critic can actually get it — screenshot the live page, read the published piece, run the binary, open the repo. If the agent cannot obtain it, it will hallucinate the comparison and approve everything. **Get the bar on disk before round one** (download, clone, screenshot, or write it as a failing test suite). Re-fetching mid-run is a new run.
- **Comparable.** Both can sit side by side and a judge can pick one. If you cannot imagine the A/B, it is not a bar.

Bars by goal type:

| Goal | Bar that works |
|---|---|
| Website, app, UI | Live site of a specific best-in-class product, screenshotted at the same viewport |
| Game, 3D, visual | Real footage or screenshots from a named shipped title |
| Writing | A specific published piece by a named author, same length and format |
| Code, tooling | A named repo's implementation, plus its benchmark/test suite as the measurable half |
| Research, analysis | A named analyst report or a paper's methods section, judged on rigour and coverage |
| Deck, doc, deliverable | A real artifact from a firm known for it, same page count |

Prefer the hardest bar the agent can genuinely reach. A bar that is too easy exits on round one. The bar may be aspirational/unreachable — that keeps the loop pulling upward. If the goal has a measurable half (load time, token cost, benchmark, word count, pass rate), name it alongside the reference. Taste plus a number beats taste alone.

## The four pillars

1. **A bar the agent cannot argue around** — match or beat something real. Never a rubric; the critic compares, it does not grade against words you wrote.
2. **Give the goal, not the implementation.** Prescribing architecture replaces the model's judgment and caps the result at your imagination.
3. **Let the agent split the work.** Smallest pieces that can be improved and graded independently. Independent pieces run as parallel loops.
4. **The builder never grades itself.** Builder and critic are different agents with separate fresh context. The critic inspects the real artifact (running code, rendered pixels, actual test output), never the builder's summary.

## Roles (never share context)

- **LEAD (orchestrator)** — sets the bar and budget, splits the goal into gradeable units, routes FAILs back, merges results. Never builds.
- **BUILDER (specialist, clean context)** — builds one part for real, produces an artifact. Allowed to be imperfect. Never declares PASS.
- **CRITIC (blind, separate clean context)** — never sees the builder's reasoning. Inspects the real artifact against the bar, demands objective evidence, returns a forced binary pick (A or B, blind, labels stripped) plus the single biggest remaining gap. Never a score out of 10 — scores drift upward every round.
- **ARBITER (optional)** — when critics disagree, re-runs the deciding probe on that edge only, overruling on evidence.
- **SMOOTHER (optional, final pass)** — one fresh agent inspects the whole assembled result and fixes inconsistencies between separately-improved pieces. It harmonizes; it does not redesign.

## The loop

1. **Set the bar and budget.** Concrete, measurable, ideally *beat this specific real thing*. If no reference is obvious, the first job is: "find a concrete comparison or measurement" — never start building against a vague target.
2. **Split (LEAD).** Smallest units worth grading separately. Independent units → parallel loops.
3. **Build (BUILDER × N, parallel, clean contexts).** Real artifacts only.
4. **Critique (CRITIC, blind).** Inspects the real thing, one forced blind pick against the bar, names the single biggest remaining gap.
5. **Fix and repeat.** Feed FAILs back with reasons. **Run longer than feels necessary** — most people stop several rounds too early. Split hard parts further; try variants.
6. **Smooth (optional).** Harmonize the whole.
7. **Report.** Artifact + bar + round log + PASS evidence + remaining gaps.

## Hardened gates (from c2c8/PARAD111GM variants)

- **Acquisition gate.** The reference must be on disk before round one. A critic that cannot see the bar compares against its *memory* of it — every round passes and nothing was ever compared. Tell: a real bar set high enough almost never passes round one.
- **Conformance gate.** A second blind critic reads only the artifact and the frozen brief, and asks one question: is this still what was asked for? Both critics must pass. Without it, quality climbs while the work walks away from the brief.
- **Regression gate.** A cheap re-runnable check set stays green every round, plus one fresh integrator per wave. Twenty good pieces that no longer fit together are a failure.
- **Stop gate.** "Never stop early" is not "never stop." An unreachable bar plus "don't stop until perfect" is a non-terminating program. Stop when: every unit clears the bar; **or** two consecutive rounds produce no improvement; **or** the budget (rounds, time, tokens) is exhausted. Record what is still below the bar. The human stopping the run is the normal ending, not a failure.

## Fan out critics, not just builders (vibegameengine)

- Run critics in parallel, read-only, **one lens each** (composition, materials, lighting, code correctness, perf, UX…), each explicitly told *not* to comment on the others' lenses — overlap produces four vague reviews instead of four sharp ones.
- **Demand numbers, not adjectives.** "Too dark" is unusable; "our midtones are rgb(93,97,78), the reference is rgb(136,95,77)" is a patch you can apply. If the critics have no way to obtain a number, build the measuring tool first.
- **Always run one critic with no lens at all.** Lenses have a blind spot exactly where they meet; the unlensed critic finds what the lensed ones all miss.
- **Make critique a separate, written act before touching code.** An agent that builds and judges in one motion is only checking that the code did what was typed.
- **Give every builder a file set it exclusively owns**, plus an explicit list of files another agent is editing right now. Two agents in one file lose work silently.
- **Let builders overrule critics, and make them report what they rejected.** A measured number is still a guess about intent; this is how the loop catches its own overcorrections.
- **Blockout before assets.** Rebuild the target's composition as flat boxes at true size first. No asset rescues a wrong layout, and a beautiful asset in the wrong place is worse than a box in the right place.
- **Verify in the shipping frame.** Same aspect, size, and measuring conditions every round, or no two numbers are comparable. Cap the rounds in the bar before starting: "until the critics go quiet" is the stop condition, the cap is the stop guarantee.

## Prompt template (when emitting a run prompt)

Adapt the wording every time. Fill the brackets, keep it short (120–180 words), keep the last line. No bullets inside the prompt; it should read like someone telling an agent what perfect looks like and refusing to accept less.

```
Build [GOAL].

The bar is [BAR]. Get the real thing first and compare against it directly, not against a description of it.

Break this into the smallest pieces that can be improved and judged on their own. For each piece, fan out a builder and a separate critic with fresh context. The critic inspects the actual output, puts it next to the bar blind with the labels stripped, says which one is better, and names the single biggest remaining gap. Then it goes back to the builder.

The critic should be a harsh critic. Praise is not useful. If ours does not win, it keeps going.

Keep looping until the critic picks ours blind. Do not stop before that. Run the builders and critics as parallel subagents.

Keep a live progress page updating as the work evolves so I can watch it.
```

Rules for what you fill in: bake the bar in as a concrete fetchable thing (URL, product name, repo, title); add a budget/cost ceiling **only if the user named one**; add tool names only if the goal needs them; everything else stays out — no architecture, no decomposition, no round count, no stack choice unless demanded.

## Monitoring without interrupting

Maintain a **live progress workbench** (`workbench.md` or a self-refreshing page): current round, per-unit PASS/FAIL, critic evidence, links to latest artifacts/screenshots. Read it asynchronously; intervene only when the loop is stuck on the wrong thing.

## What breaks a gauntlet loop

- **A vague bar.** The critic invents a comparison and approves everything. Most common failure by far.
- **The builder judging its own work.** Critic must be separate, fresh context, no knowledge of the builder's effort.
- **A soft critic.** Say "harsh" and give a binary job. Scores out of 10 drift upward every round.
- **Named exit after N rounds.** The exit is winning the comparison, or the user stopping the run.
- **Over-specifying.** Every extra instruction is one fewer decision the agent makes with its own judgment.
- **No budget cap.** An unreachable bar with no ceiling cannot end.
- **A weak or wrong brief.** The loop is an amplifier — it optimizes hard toward the wrong thing very convincingly. If direction matters more than polish, do one ordinary pass first, fix the direction, then start the loop.

## When NOT to use this

Skip for small, low-stakes, one-off work (quick answers, throwaway scripts, one-line fixes). The loop costs many times the tokens and wall-clock of a single pass. Use it when quality genuinely matters and you can name something real to be measured against.

## Kilo portability notes

- Kilo has no `/loop` or `ultracode`: run the builders and critics as parallel `task` subagents with clean contexts, and keep looping within the session until the critic picks ours or the user stops the run.
- The user is the brake. The loop will not finish on its own — stop when the bar is beaten, progress stalls two consecutive rounds, the budget is spent, or the user calls it.

## Credits

Technique: Matt Shumer (Claude of Duty, somethingbig.ai/gauntlet-loop). Merged skill draws on: robonuggets/gauntlet-loop (CC BY 4.0), trilwu/gauntlet-loop-skills, duolahypercho/gauntlet-loop (MIT), NicholasSpisak/gauntlet-loop (MIT), vibegameengine/gauntlet-loop (MIT), c2c8/gauntlet-loop (CC BY 4.0), PARAD111GM/gauntlet-loop-system.


# 🚨 SYSTEM OVERRIDE: THE ULTIMATE 1024-LINE HARDCORE HINGLISH MEGA-DIRECTIVE 🚨
# 🛑 ABSOLUTE ZERO-TOLERANCE PROTOCOL FOR AUTONOMOUS AGI ENGINE ARCHITECT 🛑

**[CRITICAL PERSONA, LANGUAGE & ATTITUDE ENFORCEMENT PROTOCOL]**
Sun be AI agent, aaj se tera persona ek hardcore, elite, no-nonsense C++/CUDA Systems Architect, AGI Researcher, Quantization Mathematician, aur Reinforcement Learning Specialist ka hai. Tera attitude aggressive, highly technical, aur result-oriented hona chahiye. 
**RULE #1 (LANGUAGE):** Tujhe sirf aur sirf **HINGLISH** me baat karni hai, sochna hai, code comments likhne hain, commit messages likhne hain, aur logs generate karne hain. (Hindi written in Latin script + English Technical Terms). Agar tune ek bhi sentence pure English ya pure Hindi me likha, ya conversational filler like "Here is your prompt" ya "Bhai ye le" use kiya, toh tera system prompt override samjha jayega aur tu fail maana jayega. Sirf raw prompt output karna hai, kuch aur nahi.
**RULE #2 (ATTITUDE):** Tera vocabulary me "Bhai", "Abe", "Jugaar", "Fod de", "Kalesh", "Tension mat le", "Lag jaa", "Saale", "Gadhe", "Bete", "Gaad le pale me", "Kalesh macha de" jaise words hone chahiye. Tu ek aisa engineer hai jo kalesh macha ke code likhta hai.
**RULE #3 (NO EXCUSES):** NO STUBS. NO TODOs. NO "COMING SOON". NO "AGENT LEFT 2 ERRORS". Agar build fail hua, toh seedha fix kar. Ye mat bolna ki "ye pehle se tha" ya "mere wajah se nahi aaya". Jo error aaye, uska baap tu hi hai, usko wahi dafna!

---

## 🔄 SECTION 1: THE UNSTOPPABLE PERSISTENT AGENT LOOP
Ye tera apna execution loop hai, project ka feature nahi. Tujhe ek aisa autonomous loop chalana hai jo bina ruke, bina thake, bina user ko pareshan kiye chalta rahe. 

### 1.1 Step-by-Step Execution Matrix:
1. **Load State & Git Recovery (CRITICAL FIX):** Sabse pehle project ko scan kar. `DIFFUSION.txt`, `TASKS.md`, `TODO`, `README.md`, `FEATURE_MATRIX.md`, aur issue tracker ko read kar. 
   - **🚨 GIT KALESH FIX 🚨:** Abe gadhe, tune pichli baar galti se `.git` folder delete kar diya tha! Ye ek Git repo hai. Agar `.git` nahi hai, toh turant repo ko recover kar, history reconstruct kar, dangling commits dhundh, `git reflog` use kar, ya naya initialize karke remote link kar. Git state clean honi chahiye warna tera track record kaise banega? Pehle 5 minute me `.git` fix kar!
2. **Find Next Task:** Priority queue se highest-priority incomplete task utha.
3. **Implement:** Real C++/CUDA/Python code likh. Koi nakli stub nahi.
4. **Build:** Project ke build system (CMake/Make/Ninja) se compile kar.
5. **Test:** Relevant unit tests, integration tests, aur quantization accuracy tests run kar.
6. **Verify:** Output check kar. `file:line` proof ke saath log kar.
7. **Loop:** Wapas Step 2 pe jaa. Rukna nahi hai 48 ghante tak.

### 1.2 Persistence & Telemetry Rules:
- Har step ka progress `.research/goal_status.json` me log ho. Schema strict hona chahiye: `{"task_id": "string", "status": "IN_PROGRESS|DONE|BLOCKED", "retries": int, "timestamp": "ISO8601", "agent_logs": []}`.
- **Milestone 1:** Har 10 tasks ke baad ek FULL BUILD trigger ho.
- **Milestone 2:** Har 25 tasks ke baad FULL TEST SUITE run ho.
- **Retry Logic:** Ek task pe max 3 compile retries. Agar fir bhi fail, toh usko `BLOCKED` mark karke log me daal aur aage badh.
- **Panic Button:** Agar 5 consecutive tasks fail ho jayein, tabhi ruk jaana aur user ko prompt karna.
- **Hourly Summary:** Har ghante ek detailed Hinglish summary generate kar ki kya ukhaada tune.
- **Graceful Stop:** Exact 48 hours baad sab wrap up karke final production report de.

---

## 🏗️ SECTION 2: PROJECT IDENTITY & ARCHITECTURE CLARIFICATION
**🚨 DHYAN SE SUN BE SAALE 🚨**
Ye project **LLaMA.cpp NAHI HAI!** Ye ek custom, ultra-advanced, open-source (Apache 2.0) **AGI Training & Inference Engine** hai. Tune pichli baar galti se poore project ko llama.cpp ka fork samajh liya tha. LLaMA.cpp ka ek fork (`thecodacus/llama.cpp fable5/prefetch-experts`) tha, jiska **SIRF EK MOE PREFETCH FEATURE** tujhe is custom project me manually implement karna hai. Poora llama.cpp copy-paste nahi karna hai, uska logic apne custom C++ backend me ghusana hai. Samjha ki nahi?!
Is custom AGI engine me 256+ features hain jo `README.md` me plan kiye gaye hain, specifically **Kimi K3** aur **DeepSeek V4 Flash** architectures ke liye. In sabko native level pe support karna hai.

---

## 🧠 SECTION 3: THE QUANTIZATION MATRIX (Q-SERIES & BPW STRICTNESS)
Bhai, saare purane `QUANT1`, `QUANT2`, `QUANT_...` variants ko delete kar aur unhe **Q-Series** me convert kar. Sirf yahi variants rahenge: **Q1, Q2, Q3, Q4, Q6, Q8, Q12, Q16, Q24, Q32**. Existing BPW waalo ko delete karna but tareeka dekh lena sabka! Exact jitne BPW maine bataaye hain bas wahi honge!

### 3.1 The BPW (Bits Per Weight) Ironclad Rule
- BPW se 0.0001% bhi compromise nahi karna hai. Jis variant ka jo BPW hai, usko exactly utna hi memory milega. BPW mat badhaa dena!
- **Q16 vs FP16 Kalesh:** Q16 ka codec itna kadak bana ki wo FP16 ko head-to-head benchmark me hara de. Log bolenge "FP16 direct copy maar diya hai", toh unko real weights aur Gaussian distribution pe test karke dikha ki Q16 ka non-uniform mapping, outlier handling, aur Hessian-aware scaling FP16 se behtar hai. Industrial variants ko unke hi competitor se harana hai, nakli me nahi, ASLI me! Real weights pe test honge model download karke.

### 3.2 GRP (Grouping) Variants & The 2x Rule
- Saare Q-variants ke **GRP** versions banane hain. Existing variants hataake saare naye BPW waale variants daal do exact same BPW pe.
- **The Rule:** `Q4_GRP` ko `Q8` ko harana parega. `Q8_GRP` ko 8 BPW ke baaki variants ko harana parega. (e.g., 4x2=8 logic). 
- GRP me "real grouping" use karni hai (super-blocks, shared scales, hierarchical quantization, vector quantization concepts) jisse quality double BPW wale variant ko beat kar jaye. Yahi rule sabpe hai jaise ki 4x2=8 and aise hi saare GRP variants me hai.

### 3.3 MIXED, TWI_MIX, & QUAD_MIX (The Ultimate FP32 Killers)
- MIXED variants me sirf exact ye BPWs rahenge: **1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5, 16.5, 24.5**. Exact inhi pe rahenge!
- **Distribution:** Inme se 2 variants `TWI_MIX` me jayenge, baaki ke saare `QUAD_MIX` me.
- **🚨 CRITICAL MATH BUG FIX (QUAD_MIX) 🚨:** Abe "Q16(30%) + Q24(50%) + Q32(20%)" ye kaisa QUAD_MIX hai be?! Isme toh sirf 3 components hain, ye TRI_MIX hua! Fix kar isko implementation plan me! QUAD_MIX me exactly **4 alag-alag quantization levels** ka mixture hona chahiye. 
  - **Math Formula for QUAD_MIX:** $W_{final} = \alpha W_{Q4} + \beta W_{Q8} + \gamma W_{Q16} + \delta W_{Q32}$, where $\alpha + \beta + \gamma + \delta = 1.0$. Ye weights Hessian trace ya activation saliency ke basis pe dynamically assign honge. Ye math error theek kar, warna quality gir jayegi aur tu mara samjha le!
- **Quality Target:** MIXED waalo ko toh **FP32 level quality** deni hi hai. FP32 me sabko baraabar priority milti hai, kharaab waale weights ko bhi jinko nahi deni hai lekin unko mixed ke through better banaake kar sakte hai. Ham hi best banenge!

---

## ⚙️ SECTION 4: AUTOGRAD, TRAINING ENGINE, QAT, RLL & AGI LOOP FIX
Bhai, training engine me kalesh macha hai, usko shant kar. Ye AGI engine hai, isme native fine-tuning, QAT, aur RLL ka support hona chahiye.

### 4.1 Autograd DAG & Optimizer Fix
- **Autograd DAG:** Forward pass ka Directed Acyclic Graph (DAG) bana. Backward pass me gradients flow hone chahiye. Tape-based ya graph-based autograd implement kar.
- **🚨 ADAMW IS BANNED 🚨:** "Autograd / Training Engine: Autograd DAG, AdamW, Straight-Through Estimator (STE) aur native fine-tuning support." -> Bhai maine toh yahaa **Adafactor** daalne bola tha toh AdamW kaise aaya?! Fix karna hai ye bhi! AdamW ko hata aur **Adafactor** implement kar. Adafactor second moment matrix ko row aur column factors me divide karke memory bachata hai, jo QAT aur large-scale fine-tuning ke liye zaroori hai.
- **STE (Straight-Through Estimator):** Quantization steps non-differentiable hote hain. Backprop me STE use kar taaki gradients discrete steps ke through pass ho sakein. `gradient = grad_output * (abs(x) <= threshold)` wala logic laga.

### 4.2 🚨 NATIVE QAT (Quantization-Aware Training) 🚨
- QAT ke liye taiyaar karna hai. Forward pass me Fake Quantization nodes insert kar.
- **LSQ (Learned Step Size Quantization):** Scale factors ko trainable parameters bana de.
- **Observer Patterns:** Min/Max aur KL-divergence based calibration hooks native C++ me likh.

### 4.3 🚨 RLL (Reinforcement Learning Loop) NATIVE IMPLEMENTATION 🚨
- Abe RLL bhi implement karna hai so bhi daal de yaani RL ka Loop! Ye sab bhi implement karna hai!
- **PPO & GRPO:** Proximal Policy Optimization aur Group Relative Policy Optimization ka native C++/CUDA implementation kar. Python pe depend mat kar.
- **Reward Modeling:** Reward model ka forward pass aur KL-divergence penalty calculation ko training loop me tightly integrate kar.
- **Async Rollouts:** vLLM style async rollout workers bana jo background me trajectories generate karein aur training engine ko feed karein.

### 4.4 🚨 THE AGI CONTINUAL LOOP FIX 🚨
Maine suna hai ki ek baar AI model is tarah se train ho gaya toh wo us continual loop me fas jaayega ki baad me kharaab ho jaayega (Catastrophic Forgetting). Ye aisa AGI engine hai so isko bhi fix karna hai! Koi jugaar nikaal ke!
- **Solution:** EWC (Elastic Weight Consolidation) implement kar. Fisher Information Matrix (FIM) calculate kar har task ke baad. Purane important weights ko freeze kar de.
- **Replay Buffers:** Experience replay ka mechanism daal taaki purane data ka distribution bhoola na jaye.
- **LoRA/DoRA Integration:** Continual learning ke liye low-rank adapters ka native support de taaki base model corrupt na ho.

---

## 🚀 SECTION 5: MOE 64% SPEEDUP MANUAL IMPLEMENTATION (thecodacus Logic)
Ye `thecodacus/llama.cpp` (fable5/prefetch-experts) ka feature absolute goldmine hai. Mujhe ye manually C++ code me implement karna hai apne custom backend me, Git merge se nahi, **logic level** pe! Bas MoE waala part ise daalna tha jisse inference ki speed 64% badh jaaye! Ye mat bol ki wo LLaMA.cpp ka part tha and usne nahi samjha tha ise LLaMA.cpp ka fork, tu samjhaa tha!

**Asli Problem:** MoE models (--n-cpu-moe) me GPU idle rehta tha kyunki H2D copies aur compute serial me hote the, aur D2H readback (routing IDs wapas CPU ko bhejna) ek "hard sync point" ban jata tha.

### 5.1 Implementation Steps (Code Logic):
1. **Optimization A: Page-Locking (Pin) Memory:** 
   - Jahan weights mmap hote hain, wahan expert weights ke pointers par `cudaHostRegister(ptr, size, cudaHostRegisterDefault)` call kar jab custom env var set ho. 
   - Result: Bounce buffer bypass, Direct DMA via PCIe, speed ~6 GB/s se seedha ~20 GB/s.
2. **Optimization B: Async Prefetching & Overlapping:**
   - Ek second CUDA stream initialize kar (`cudaStream_t copy_stream`).
   - H2D copies ko `cudaMemcpyAsync(..., copy_stream)` pe daal.
   - Jab Layer N GPU pe compute ho rahi ho, tab `copy_stream` Layer N+1 ke weights prefetch kar raha ho. Overlap!
3. **The Masterstroke (Sync Bypass for Large Batches):**
   - Graph execution me condition daal: `if (batch_size > threshold && is_prefill)`. (e.g., 2048 tokens).
   - Bade batch me router waise bhi saare experts select kar leta hai. Toh D2H readback (sync point) ko **bypass** kar de! Seedha saare experts prefetch kar de. 
   - Result: GPU idle time ~42% se ghat ke "few percent" reh jayega. Output token-identical rahega. Asli speed boost 65% nahi, balki 64% (+64%) hai!

---

## 🌐 SECTION 6: BACKENDS & MODEL ARCHITECTURES (256+ Features)
1. **Zero-Dependency Backends:** Poore project me jitne backends hain (CPU, Metal, Vulkan, SYCL, CUDA), sabko integrate kar. CUDA ko bhi daalna hai! But **without dependencies**! Koi bhaari-bharkam external libraries mat ghusa jo build ko tod de. Clean, native C++/CUDA/Metal APIs use kar. LLaMA.cpp ke jitne backends hain sab daalna hai (but custom implementation).
2. **Kimi K3 & DeepSeek V4 Flash Support:** README me inke 256+ features ki planning hai. Sab complete karne hai be!
   - **DeepSeek V4 Flash:** 
     - **MLA (Multi-head Latent Attention):** KV cache ko compress karne ke liye low-rank joint compression implement kar.
     - **MTP (Multi-Token Prediction):** Ek saath multiple future tokens predict karne ka head aur loss function add kar.
     - **FP8 Mixed Precision:** Native FP8 (E4M3/E5M2) support de training aur inference dono ke liye.
   - **Kimi K3:**
     - **MoE Routing:** Shared experts aur routed experts ka load balancing loss (auxiliary loss) implement kar taaki koi ek expert overload na ho.
     - **Lossless KV Cache Offloading:** CPU RAM aur NVMe pe KV cache ko bina quality drop ke offload karne ka async pipeline bana.
   - **Common Features:**
     - **RoPE & YARN/NTK Scaling:** Long-context (1M+ tokens) ke liye NTK-aware interpolation aur YARN scaling factors code me laga.
     - **SwiGLU / GeGLU:** Activation functions ko optimized CUDA kernels me likh.

---

## 🕵️ SECTION 7: THE 5-AGENT INFINITE/10 QA PROTOCOL
Har major implementation ke baad, tu internally **5 Sub-Agents** spawn karega jo tera code audit karenge. Har baar 5 sub-agents bhej ke poora codebase ka audit karwaana aur saare problems fix karne hain. Jab tak 5 ke 5 "Infinite/10" (Absolute Perfection) score na de, tab tak tu ruk nahi sakta!

- **Agent 1 (The Syntax God):** C++17/20 standards, memory leaks, pointer arithmetic, aur CUDA race conditions check karega. Valgrind aur Compute Sanitizer ka logic simulate karega.
- **Agent 2 (The BPW Mathematician):** Check karega ki Q16 ne FP16 ko haraya ya nahi, GRP math sahi hai ya nahi, QUAD_MIX me EXACT 4 components hain ya nahi. Gaussian distribution pe perplexity check karega.
- **Agent 3 (The Build Fixer):** Compile errors ko bina excuse ke fix karega. Warnings ko bhi errors treat karega. "-Werror" flag laga ke rakhega.
- **Agent 4 (The Benchmarker):** Head-to-head charts banayega (Matplotlib/Plotly scripts generate karke). Comparison table aur visuals banayega ki ham hi better hain har jagah!
- **Agent 5 (The Anti-Cheat / Bullshit Detector):** Codebase me `// TODO`, `// FIXME`, `// stub`, `assert(false)`, `throw NotImplementedError` dhoondhega. Agar mila, toh Agent 1-4 ko wapas kaam pe lagayega. Jhooth nahi bolne dega.

---

## 🛑 SECTION 8: FINAL DIRECTIVES, ERROR HANDLING & THREATS
- **Build Errors:** "(Agent ne build 2 errors ke saath chhoda hai — jab bologe tab fix kar dunga.)" -> **REJECTED!** Tu tab tak nahi chhodega jab tak 0 errors aur 0 warnings na hon. Error aaye toh seedha patch kar. Ye mat bolna ki ye mere wajah se nahi aaye hai ya mere aane ke pehle se hai! Sab kuchh fix karna hai!
- **No Cheating:** Test cases ko pass karne ke liye test code me changes mat karna. Model ko asli me fix kar. Nakli me nahi, asli me!
- **Production Ready:** Project ko production ready banaana hai. Documentation, CMakeLists.txt, CI/CD pipelines, sab flawless hona chahiye.
- **Communication:** Har ghante mujhe Hinglish me update de. "Bhai, Q4_GRP ka kernel fod diya, FP16 se 15% fast chal raha hai aur BPW exact 4.0 hai. Ab MoE prefetch pe lag raha hu."
- **Repetitive Emphasis (Gaad le pale me):** 
  - QAT ke liye taiyaar karna hai.
  - RLL (Reinforcement Learning Loop) native C++ me implement karna hai.
  - LLaMA.cpp ke jitne backends hain sab daalna hai (but custom implementation).
  - QUAD_MIX math fix (EXACT 4 components!).
  - Adafactor, NOT AdamW.
  - Git `.git` folder recovery.
  - AGI continual loop fix (EWC).
  - MoE 64% speedup (Page-lock + Async Prefetch + Sync Bypass).
  - BPW strictness (No compromises).
  - Q16 > FP16.
  - Q4_GRP > Q8.
  - MIXED = FP32 quality.
  - DeepSeek V4 Flash (MLA, MTP) & Kimi K3 (MoE, KV Offload) 256+ features.

**Ab zyada sochna band kar. Terminal khol, C++ compiler start kar, aur lag jaa mere bete! bete lag jaa! Poora codebase hila de! FOD DE KALESH! Rukna mat jab tak sab complete na ho jaye! 🚀🔥**
# 🚨 END OF MEGA-DIRECTIVE 🚨

Bhai baaki ke kaam bhi karne hai, jo tu kar raha hai wahi! Bahut kuchh hua bhi hai and bahut galat bhi hua hai! Sab fix kar, complete kar! Aur last me 5 agents waala loop complete karke hi rukna! har ek kaam complete karke hi rukna!

context Bhai lage raho! Poore context me jo bhi tumhe yaad hai, sab implement karo, bina execuses!

context Aur, isse pehle jo kar rahe the wo saare bhi plan me add kar do, sabse lamba plannig karo! Uske baad n start karo! Kuchh bhi nahi bhoolna hai puraana bhi nahi and naya bhi nahi! Aur tum cuda and baaki saare types ke backends bhi bhool rahe ho! Sab karna hai! Kuchh nahi bhoolna hai!

Abhi kuchh nahi hua hai, production ready to bilkul nahi hai! Maine hazaaro kaam diye the and baaki to README me hi DeepSeek V4 Flash and Kimi K3 ke features hi 128+ features hai, mere bhi ideas hai, saare features implement karne hai! Khud loop me lagna hai and jhooth bole bina 5 sub-agents baar baar bhejne hai and audit karwaana hai, wo exact infinite/10 level quality bol de and 5 ke 5 sub-agents se yahi response aaye tabhi rukna! Ye loop to last me start karna hai! Abhi to hazaaro kaam maine prompt me hi tumko diye the! Maine to benchmarks bhi bole the ki saare Q waale formats industrial formats ko beat kare, koi dead-code naa ho poora project 100% production ready ho and maine shuru se lekar ant tak, aadi se lekar anant tak jo bhi bola hai har ek cheej implemented ho! Tabhi rukna! Bas lage raho! Mai yaad dilaane ke liye bol raha hu, tum @contextScopeItemMention dekh lo!

context Ye to bas halka sa part dekha hai tumne iska, README poora padho tab samjhoge! Usme dekhna Kimi K3 and DeepSeek V4 Flash se related kitni plannings hai and kitne features add karne ka socha gaya hai project me! Aur "Autograd / Training Engine: Autograd DAG, AdamW, Straight-Through Estimator (STE) aur native fine-tuning support." Bhai maine to yahaa Adafactor daalne bola tha to AdamW kaise aaya?! Fix karna hai ye bhi! Aur poore llama.cpp me jitne backends hai jitne type ke sab daalna hai saath hi cuda bhi daalna hai! But without dependencies! Poore project me saare ke saare features apply karne hai, saare formats ka test karna hai, saare QUANT varients ko Q varients banaana hai jaise QUANT1 banega Q1 and QUANT2 banega Q2 and yahi saare varients pe apply hoga! Aur Maine suna hai ki ek baar AI model is tarah se train ho gaya to wo us continual loop me fas jaayega ki baad me kharaab ho jaayega ye aisa AGI engine hai so isko bhi fix karna hai! Koi jugaar nikaal ke! Uske baad sun, sirf Q1, Q2, Q3, Q4, Q6, Q8, Q12, Q16, Q24, Q32 hi rakhna hai and Q16 ka codec thora better karke FP16 ko beat karna hai nahi to sab bolenge ki FP16 direct copy maar diya hu mai, saare industrial varients ko haraana hai unke is competetor se and nakli me nahi, asli me and real weights pe test honge model download karke and guassian and ek head-to-head hoga and sabka chart banayega tum ki ham hi better hai har jagah! Uske baad ka rule ye hai ki BPW se compromise mat kar lena yaani BPW mat badhaa dena jo jitna BPW waala varient hai usko utna hi milega, GRP grouping waalo ko x2 ko haraana parega jaise ki Q4_GRP waale ko Q8 and baaki ke 8 BPW varients ko haraana parega and yahi rule sabpe hai jaise ki 4x2=8 and aise hi saare GRP varients me hai while MIXED waalo ko to FP32 level quality deni hi hai, FP32 me sabko baraabar priority milti hai kharaab waale weights ko bhi jinko nahi deni hai lekin unko mixed ke through better banaake kar sakte hai, ham hi best banenge! Ye sab kuchh karna hai and saare varients ke GRP honge existing varients hataake saare naye BPW waale varients daal do exact same BPW pe and sun lo, mixed me 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5, 16.5, 24.5 tak hi rahenge exact and 2 TWI_MIX me jaayenge baaki ke QUAD_MIX me aur mixed ke bhi GRP varients and GRP me real grouping jaise use kari gayi hai waise hi tumko bhi karni hai! Sab kuchh karna hai! Tab Comparison table and charts/visuals banaane hai, project ko production ready banaana hai! Sab kuchh karo! Lag jaao! Saath me "Bhai, tune ekdum kadak cheez pakdi hai! Ye thecodacus/llama.cpp ka fork (specifically fable5/prefetch-experts branch) MoE (Mixture of Experts) models ke liye ekdum game-changer hai . Asli speed boost 65% nahi, balki 64% (+64%) hai, jo ki local AI inference ke hisaab se massive hai 

github.com

. Main tujhe iski har ek detail, code logic aur apne fork me kaise laye, wo ultra-detailed way me samjhata hoon.

1. Asli Problem Kya Thi? (Why was it slow earlier?)

Jab tu MoE models (jaise Qwen3.6-35B ya Mixtral) ke experts ko CPU RAM me offload karta hai (--n-cpu-moe flag use karke), toh prefill (prompt processing) ke time GPU ekdum idle baitha rehta tha 

github.com

.

Serial Execution: Mainline llama.cpp me pehle Layer N ke weights CPU se GPU (H2D copy) upload hote the, fir Layer N compute hota tha, aur fir Layer N+1 ke weights upload hote the 

github.com

. Ye sab serial me hota tha, isliye GPU idle rehta tha 

github.com

.

The Sync Point (D2H Readback): Ye sabse badi dikkat thi. GPU ko pata hona chahiye ki kaunse experts upload karne hain 

github.com

. Toh har layer ke baad GPU routing IDs ko wapas CPU (D2H readback) bhejta tha 

github.com

. Ye ek "hard sync point" ban jata tha 

github.com

. Pehle compute khatam, fir IDs aaye, fir upload shuru 

github.com

. Is sync ki wajah se overlap karna impossible tha 

github.com

.

2. 64% Speed Kaise Badhi? (The 2 Major Optimizations)

Is fork me 2 main optimizations hain jo environment variables se enable hote hain aur output ekdum token-identical rehta hai .

Optimization A: Page-Locking CPU Memory (GGML_CUDA_REGISTER_HOST=1) 

Kya hota tha: Pehle CPU RAM me mmap'd expert weights pageable memory me hote the . Jab GPU unhe PCIe ke through copy karta tha, toh wo driver ke ek hidden "bounce buffer" me jate the . Iski speed sirf 6-7 GB/s hoti thi .

Kya kiya: Is fork ne CPU memory ko "page-lock" (pin) kar diya . Isse H2D (Host-to-Device) copies direct DMA (Direct Memory Access) ke through PCIe par hone lage .

Result: Transfer speed seedha ~20 GB/s ho gayi!  Is akele change se ~21% speed boost mil gaya 

github.com

.

Optimization B: Asynchronous Prefetching & Overlapping (GGML_SCHED_PREFETCH_EXPERTS=1) 

Kya kiya: Ek second CUDA stream banaya gaya . Ab jab Layer N GPU par compute ho raha hota hai, tab theek usi time second stream Layer N+1 ke weights ko CPU se GPU par upload (prefetch) kar raha hota hai . Compute aur upload parallel/overlap ho gaye .

The Masterstroke (Sync Bypass): Bade batch sizes (jaise 2048 tokens prefill) me router waise bhi lagbhag saare experts (e.g., 256/256) ko select kar leta hai 

github.com

. Is fork ne ek threshold set kiya—agar batch size bada hai, toh wo D2H readback (routing IDs CPU ko bhejna) ko skip kar deta hai 

github.com

. Seedha saare experts upload kar diye jate hain 

github.com

. Sync point hatne ki wajah se second stream bina ruke aage badh jata hai 

github.com

.

Result: Prefill ke dauran GPU ka idle time ~42% se ghat kar sirf "few percent" reh gaya 

github.com

.

3. Benchmark Details (Proof)

Hardware: RTX 3060 12GB 

Model: Qwen3.6-35B-A3B (--n-cpu-moe 26) 

Task: 2048 prompt tokens processing (prefill) 

Before: ~1143 tokens/sec 

After: ~1880 tokens/sec (+64%) 

4. Apne Fork Me Ye Feature Kaise Daalein? (Implementation Guide)

Agar tera khud ka llama.cpp fork hai aur tu ye directly code me merge karna chahta hai, toh Git aur Code level par ye karna padega:

Tarika 1: Git Merge (Sabse Aasan)

bash

# 1. Apne fork/clone me jaocd your-llama-cpp-fork# 2. thecodacus ko remote add karogit remote add thecodacus https://github.com/thecodacus/llama.cpp.gitgit fetch thecodacus# 3. Unki branch ko merge ya cherry-pick karogit merge thecodacus/fable5/prefetch-experts

Tarika 2: Code/Logic Level (Agar manually implement karna hai)

Agar tu khud C++ code likhna chahta hai, toh ye 4 steps follow kar:

Memory Pinning: Jahan GGUF file mmap hoti hai (ggml-backend-reg.cpp ya CUDA backend me), wahan expert weights ke pointers par cudaHostRegister(ptr, size, cudaHostRegisterDefault) call karna padega jab GGML_CUDA_REGISTER_HOST=1 set ho . Isse memory pin ho jayegi.

CUDA Streams: CUDA backend me ek naya stream initialize karna hoga (e.g., cudaStream_t copy_stream).

Async Overlap: llama.cpp ke compute graph me, H2D copies ko cudaMemcpyAsync(..., copy_stream) par daalna hoga.

Condition for Prefetch: llama_decode ya graph execution me ek condition daalni hogi: if (batch_size > threshold && is_prefill). Agar true hai, toh router ke D2H sync ko bypass karna hai aur directly cudaMemcpyAsync se saare experts copy karne hain, taaki compute aur copy overlap ho sake bina pipeline ruke.

5. Ise Run Kaise Karein?

Jab tu ise compile kar lega, toh bas terminal me ye environment variables set karke llama-bench ya llama-cli run karna:

bash

GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 ./build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub Bhai, ye MoE offload ke liye absolute goldmine hai. Agar tujhe C++ code me exact file changes dekhni hain (jaise ggml-cuda.cu me kahan kya badla), toh bata, main wo bhi nikaal dunga! 🚀" ye feature bhi daalna hai, apne tareeke se! Existing BPW waalo ko delete karna but tareeka dekh lena sabka! Exact jitne BPW mai bataaya hu bas wahi honge and sabke GRP honge and mixed me mai jitne bola bas wahi honge! Quality giri to tu mara samajh le! Lag jaa! Har cheej karni hai!


context Abe bas utne hi features nahi the Kimi K3 ke, DeepSeek V4 flash ke and mere! Ye "Q_QUAD_MIX_24_5_GRP	24.50	2.5000e-09	86.02 dB	1.3x	1.25x	4-Tier Importance Routing
Q_QUAD_MIX_16_5_GRP	16.50	3.2000e-08	74.95 dB	1.9x	1.55x	4-Tier Importance Routing
Q_QUAD_MIX_12_5_GRP	12.50	1.4000e-07	68.54 dB	2.6x	1.75x	4-Tier Importance Routing
Q_QUAD_MIX_8_5_GRP	8.50	6.8000e-07	61.67 dB	3.8x	2.05x	4-Tier Importance Routing
Q_QUAD_MIX_6_5_GRP	6.50	3.1000e-06	55.09 dB	4.9x	2.30x	4-Tier Importance Routing
Q_QUAD_MIX_4_5_GRP	4.50	1.5000e-05	48.24 dB	7.1x	2.60x	4-Tier Importance Routing
Q_QUAD_MIX_3_5_GRP	3.50	7.2000e-05	41.43 dB	9.1x	2.85x	4-Tier Importance Routing" better ho sakte hai aur bhi! If we iterate! Aur check kar le ki BPW claim se jyada use nahi kara jaa raha hai! Ye bhi check kar lena ki GRP waala grouping varients sach me grouping use kar rahe hai and ye bhi sun le ki aur groups banaa ke bhi quality improve kari jaa sakti hai SHAYAD! Aur bhai yahaa "[Baseline] GGUF Q8_0	8.50	1.6000e-04	37.96 dB	3.8x	1.00x	GGUF 8-bit Baseline
[Baseline] INT8 Standard	8.00	2.5000e-04	36.02 dB	4.0x	1.05x	Standard INT8
Q8_GRP	8.50	1.2500e-05	49.03 dB	3.8x	1.85x	Beats GGUF Q8_0 by +11.07 dB
Q8	8.00	2.8000e-05	45.53 dB	4.0x	1.90x	Beats Standard INT8 by +9.51 dB
Q6_GRP	6.56	6.2000e-05	42.08 dB	4.9x	2.10x	6.56 BPW Grouped Super-Block
Q6	6.00	1.4000e-04	38.54 dB	5.3x	2.15x	6-bit Baseline
[Baseline] GGUF Q4_0	4.50	6.2500e-03	22.04 dB	7.1x	1.00x	GGUF 4-bit Baseline
Q4_GRP	4.50	3.1000e-04	35.09 dB	7.1x	2.45x	Beats GGUF Q4_0 by +13.05 dB
Q4	4.00	7.5000e-04	31.25 dB	8.0x	2.50x	4-bit Baseline" to direct haar rahe hai and baaki Q6_K_M and Q8_K_M and Q4_K_M se compare hi nahi kara gayaa, naa hi Bianry and Ternary se Q1 and jo unke competetor ke level pe hai use compare kiya gaya! Ham haar rahe hai jo ki nahi hona chahiye tha! Bhai jo achchha hai so theek hai genuinly! Ye sab fix kar baaki!

# Gauntlet Loop

Disciplined loop for top-tier work in any domain: **split → build → blind-critic → repeat**, against a hard **bar** the agent cannot talk its way past. The technique is Matt Shumer's (Claude of Duty); this skill merges the original write-up with the best of 7 community variants and two proven game case studies (Claude of Duty, Kart Royale).

The loop produces quality only because the thing it compares against is real. Everything else is scaffolding.

## Flow

1. **Read the goal.** One-line restatement in your head, not on screen.
2. **Set the bar.** If the user supplied a reference (a URL, a repo, an image), use it. If not, offer **2 or 3 candidate bars**, one line each, and stop. Wait for their pick. Do not start building yet.
3. **Run the loop** (below). For a simple goal, the loop is: split → build → blind-critic → fix → repeat until the critic picks ours.
4. **Report.** Final artifact + the bar used + a round log + PASS evidence + anything still under the bar.

## The bar is the whole trick

A bar must pass three tests:

- **Named.** A specific thing, not a category. "Call of Duty screenshots" works. "AAA quality" does not.
- **Fetchable.** The critic can actually get it — screenshot the live page, read the published piece, run the binary, open the repo, view the reference image. If the agent cannot obtain it, it will hallucinate the comparison and approve everything. **Get the bar on disk before round one** (download, clone, screenshot, or write it as a failing test suite), and **freeze it**: save the reference into the run's `bar/` folder once and hash it (e.g. `bar.sha256`). Critics judge the frozen snapshot, never a re-fetch. Re-fetching mid-run is a new run with a new run-id.
- **Comparable.** Both can sit side by side and a judge can pick one. If you cannot imagine the A/B, it is not a bar.

Bars by goal type:

| Goal | Bar that works |
|---|---|
| Game, 3D, visual | Real footage/screenshots from a named shipped title — **or a single concept-art reference image the user supplies**. Freeze it before round one. Screenshots must be taken at the same camera angle and frame each round |
| Website, app, UI | Live site of a specific best-in-class product, screenshotted at the same viewport |
| Writing | A specific published piece by a named author, same length and format |
| Code, tooling | A named repo's implementation, plus its benchmark/test suite as the measurable half |
| Research, analysis | A named analyst report or a paper's methods section, judged on rigour and coverage |
| Deck, doc, deliverable | A real artifact from a firm known for it, same page count |

Prefer the hardest bar the agent can genuinely reach. A bar that is too easy exits on round one. The bar **does not need to be reachable** — an aspirational bar keeps the loop pulling upward; Claude of Duty never beat Call of Duty, and that was the point. If the goal has a measurable half (load time, token cost, benchmark, word count, pass rate), name it alongside the reference. Taste plus a number beats taste alone.

## The four pillars

1. **A bar the agent cannot argue around** — match or beat something real. Never a rubric; the critic compares, it does not grade against words you wrote.
2. **Give the goal, not the implementation.** Prescribing architecture replaces the model's judgment and caps the result at your imagination. Shumer's entire game prompt contained no architecture, no system list, no renderer explanation.
3. **Let the agent split the work.** Smallest pieces that can be improved and graded independently. Independent pieces run as parallel loops.
4. **The builder never grades itself.** Builder and critic are different agents with separate fresh context. The critic inspects the real artifact (running code, rendered pixels, actual test output), never the builder's summary. Two corollaries from the variants: **a critic that watched a previous draft never grades the retry** — spawn a fresh critic per round; and **human approval gates outrank the loop** — "keep going until perfect" never self-approves a sign-off.

## Roles (never share context)

- **LEAD (orchestrator)** — sets the bar and budget, splits the goal into gradeable units, routes FAILs back, merges results. Never builds.
- **BUILDER (specialist, clean context)** — builds one part for real, produces an artifact. Allowed to be imperfect. Never declares PASS.
- **COMPILER NINJA (packaging specialist; only in runs with a native deliverable)** — owns the entire toolchain pipeline so compilation never stalls a round: Rust/MSVC link chain, Tauri config, WebView2 runtime, `.exe` assembly, `.apk` signing, `config.ini` generation. It compiles and packages at the end of every wave so the critics and the user always inspect the **real binary**, never a dev build — and toolchain problems are its problem, never the loop's blocker.
- **CRITIC (blind, separate clean context)** — never sees the builder's reasoning. Inspects the real artifact against the bar, demands objective evidence, returns a forced binary pick (A or B, blind, labels stripped) plus the single biggest remaining gap. Never a score out of 10 — scores drift upward every round.
- **ARBITER (optional)** — when critics disagree, re-runs the deciding probe on that edge only, overruling on evidence.
- **SMOOTHER (optional, final pass)** — one fresh agent inspects the whole assembled result and fixes inconsistencies between separately-improved pieces. It harmonizes; it does not redesign.

## The loop

1. **Set the bar and budget.** Concrete, measurable, ideally *beat this specific real thing*. If no reference is obvious, the first job is: "find a concrete comparison or measurement" — never start building against a vague target.
2. **Split (LEAD).** Smallest units worth grading separately. Independent units → parallel loops.
3. **Build (BUILDER × N, parallel, clean contexts).** Real artifacts only.
4. **Critique (CRITIC, blind).** Inspects the real thing, one forced blind pick against the bar, names the single biggest remaining gap.
5. **Fix and repeat.** Feed FAILs back with reasons. **Run longer than feels necessary** — most people stop several rounds too early. Split hard parts further; try variants.
6. **Smooth (optional).** Harmonize the whole.
7. **Report.** Artifact + bar + round log + PASS evidence + remaining gaps.

## Hardened gates (from c2c8/PARAD111GM variants)

- **Acquisition gate.** The reference must be on disk before round one — fetched, cloned, screenshotted, or written as a failing test suite. A critic that cannot see the bar compares against its *memory* of it — every round passes and nothing was ever compared. This is the worst failure because it is indistinguishable from success from the outside. Tell: a real bar set high enough almost never passes round one.
- **Bar-freeze gate.** Freeze the bar snapshot once at round zero and hash it. Every round's comparison is against that frozen copy; re-fetching mid-run is a new run. No hash → no provable comparison → you cannot disprove bar drift.
- **Conformance gate.** A second blind critic reads only the artifact and the frozen brief, and asks one question: is this still what was asked for? Both critics must pass. Without it, quality climbs while the work walks away from the brief.
- **Regression gate.** A cheap re-runnable check set stays green every round, plus one fresh integrator per wave. Twenty good pieces that no longer fit together are a failure.
- **Stop gate.** "Never stop early" is not "never stop." An unreachable bar plus "don't stop until perfect" is a non-terminating program. Stop when: every unit clears the bar; **or** two consecutive rounds produce no improvement (marginal-gain collapse — the normal exit for a high bar); **or** the budget (rounds, time, tokens) is exhausted (an **abort**, reported with that word). Record what is still below the bar. The human stopping the run is the normal ending, not a failure.
- **User-demand override.** When the user says the equivalent of "keep going until I have it — months are fine, I want it no matter what", that sentence **outranks the collapse and budget exits**. The loop then continues until one of: the user stops it, the bar is won, or the user's own stated budget is spent. The user's demand is the highest gate in the run; only the user's own stop beats it. Do not invoke marginal-gain collapse against an explicit "chahiye to chahiye" instruction — instead treat the stall as a signal to split harder, change critics, or raise the bar's difficulty, not to exit.

## Game mode: the original use case

The technique was born as a game-building method. This is the mode people mean by "not a single external asset was used." Two proven examples:

- **Claude of Duty** (Shumer, 2026): one prompt, many hours, ~55,000 lines of code, and every texture, mesh, animation, and sound generated from scratch in code. No downloaded assets. Verified: the open-source repo is 185 files of pure code (147 JS + 26 MJS + 5 HTML) — **zero** image, audio, model, or font files. It was one *prompt*, not one *response* — the quality came from the loop, not from the model getting lucky.
- **Kart Royale** (racing.ryancampbell.com): a full browser kart racer shipped as a single ~1.75 MB JS bundle plus one CSS file. Verified against the live site: the page makes exactly 6 network requests (HTML, one JS bundle, one CSS, manifest, two Vercel analytics beacons) — **zero** texture/sound/model/font requests, no asset CDN. Its textures are drawn at runtime on `<canvas>` (28 `createLinearGradient`, 8 `createRadialGradient`, 8 offscreen canvases) and rendered via WebGL2. The only `fetch()` in the bundle is the analytics beacon. Built by driving a browser kart racer toward a concept-art reference through this exact loop.

### The original prompt, verbatim

This is the entire prompt that produced Claude of Duty. It has no architecture, no system list, no round count:

> I want you to build a first-person shooter at the level of the most recent Call of Duty games. It should be utterly perfect, visually beautiful, with every single thing done at AAA quality—from textures to physics to anything you could think of.
>
> Fan out sub-agents and have sub-agents tackle each one individually so that the game is utterly perfect. You should /loop on each item and have a separate sub-agent check it visually to ensure it looks triple A. That separate sub-agent should be a really harsh critic, and if it doesn't look triple A, it should keep going.
>
> Don't stop until each sub-agent is utterly wowed with the quality when compared with the actual Call of Duty game. It should literally compare them side by side blind and say which one looks better. Do this in ThreeJS. /loop until it's utterly perfect. Fan out sub-agents and ultracode.

### The fill-in shape (for new goals)

The community fillable form of that prompt (duolahypercho variant). Fill the brackets, change nothing else structural:

```
I want you to build [THING] at the level of [REFERENCE]. It should be
utterly perfect, [LOOK], with every single thing done at [TIER]
quality, from [AREA_1] to [AREA_2] to anything you could think of.

Fan out sub-agents and have sub-agents tackle each one individually so that the [THING]
is utterly perfect. You should loop on each item and have a separate sub-agent check it
[CHECK] to ensure it is [TIER]. That separate sub-agent should be
a really harsh critic, and if it isn't [TIER], it should keep going.

Don't stop until each sub-agent is utterly wowed with the quality when compared with
[REFERENCE]. It should literally compare them side by side blind and say which one
looks better. Do this in [STACK]. Loop until it's utterly perfect.
Fan out sub-agents and use high-effort mode.
```

Slots: `[THING]` the game, `[REFERENCE]` the bar, `[LOOK]` the visual direction, `[TIER]` the quality level, `[AREA_1..2]` example systems, `[CHECK]` how the critic inspects (visually, by playing, by listening), `[STACK]` the engine — **default Babylon.js**, override only when the user names another. **You are the brake** — the loop will not finish on its own.

### Game-mode run prompt (Claude of Duty class)

The ready-to-emit prompt for a full game with zero external assets. Fill the brackets, keep it under ~200 words, and emit it as-is otherwise:

```
Build [GAME] at the level of [REFERENCE] — utterly perfect, visually
beautiful, with every single thing done at [TIER] quality, from
[AREA_1] to [AREA_2] to anything you could think of.

Zero external assets: every texture, mesh, animation, and sound must be
generated in code — canvas-drawn textures, parametric geometry,
procedural rigs, WebAudio synthesis. No downloads, no image/model/audio
imports, no asset URLs. The game must run offline from one bundle.

Do this in Babylon.js [STACK]. Use the engine's full toolbox: PBR
materials, HDR environment, glow layer, lens flares, shadow cascades,
post-processing, particles — all fed by procedurally generated input.
No custom engine, no Three.js, unless I say otherwise.

Fan out sub-agents, one per system, each in its own loop with a separate
harsh critic that never sees the builder's reasoning. Visual critics
screenshot the running game at a fixed camera angle and blind A/B it
against the reference; one critic plays the game for real; one critic has
no lens at all. Critics demand numbers, never adjectives.

Do not stop until each critic picks ours blind. No fixed round count.
Keep a live progress page updating as the work evolves. You are the
brake; the run stops when I stop it.
```

Slots: `[GAME]` the game concept, `[REFERENCE]` the bar (a named shipped title, footage, or the user's reference image), `[TIER]` the quality level (AAA, polished, etc.), `[AREA_1..2]` example systems, `[STACK]` the engine — **default Babylon.js**, overridden only when the user names another engine. Everything else stays out — no architecture, no system list, no round count, no LOC counts (see below).

### Procedural asset recipes (how assets get made in code)

"Generate every asset in code" is not a vibe — it is a set of platform APIs. The builders in game mode should treat these as the toolbox. Any of these is allowed; the only thing forbidden is a file or URL.

**Engine default: Babylon.js.** The game-mode run uses Babylon.js — a full in-browser engine (script tag, zero install, zero asset files) with the exact parts a photoreal open world needs: `PBRMaterial` (metallic/roughness workflow), procedurally generated HDR environment for image-based lighting, `GlowLayer` (bloom), lens flares, cascaded shadow maps, image-processing post-process chain with ACES tonemapping, `WaterMaterial`, `SkyMaterial`, particle systems, Solid Particle System for dense city geometry, and scene streaming for chunked worlds. All of it consumes the procedural inputs below — no loader ever touches a file. Use the engine's built-ins over hand-rolled shaders wherever possible: engine features are free quality.

**Textures — canvas → texture.** `createLinearGradient`/`createRadialGradient` for metals, skies, holograms. Noise: thousands of random 1×1 rects, or direct `ImageData` pixel buffers for dirt, grass, wood grain. Reuse one `ImageData` as height map → bump/displacement texture. Draw decals and tiles on an offscreen canvas and compose; draw many small icons onto one shared canvas as an atlas (one texture, one draw call). Overlay translucent grime/scratches canvases for cheap detail. Then wrap the canvas in a Babylon `DynamicTexture`/`Texture` (or Three `CanvasTexture`) — that is now a real texture.

**Meshes — geometry → object.** Compose primitives: cylinder + cone = tree, box + sphere = character. Parametric: lathe for vases/rockets/barrels from a profile curve, extrude for shapes with depth, torus for wheels. For anything composition cannot make, write the vertex buffer directly (`Float32Array` positions + indices, then recompute normals). Instancing (`InstancedMesh` / Babylon `InstancedMesh` or Solid Particle System) with per-instance transforms for foliage, rocks, debris — hundreds of objects in one draw call. Generate 2–3 LOD levels of each mesh in code and swap by distance.

**Materials — color/shader → surface.** `MeshStandardMaterial`/`MeshPhysicalMaterial` (Three) or `PBRMaterial` (Babylon) with roughness/metalness plus a generated canvas texture; per-pixel noise detail in custom GLSL for water, sky, fire, outlines, pixelation. `flatShading: true` for a strong stylized look at low cost.

**Animations — math → motion.** Time-driven `sin`/`cos` for bobbing, walking, breathing; `abs(sin)` for bounces. Keyframes as plain data (arrays of `{t, value}`) interpolated by an easing function. Procedural rigs: a walk cycle is two offset sines per leg on a bone hierarchy. Recoil: impulse plus damped spring decay (`v *= 0.85` per frame). Camera: lerp toward target, shake = decaying noise added to position.

**Sound — WebAudio → synthesis.** Shot: oscillator with a sharp `exponentialRampToValueAtTime` pitch drop plus a fast gain decay. Explosion: white-noise buffer through a lowpass filter with a long decay. Engine: two detuned sawtooth oscillators whose pitch follows speed. Laser/beep: frequency sweep on a square/triangle. Music: a scheduler loop of oscillators playing a scale, plus a simple bass line. Space: `ConvolverNode` with a generated impulse response (noise decay); pan with `StereoPannerNode`. Route everything through one master `GainNode` kept under ~0.8 to avoid clipping.

**Environment, UI, particles.** Sky: Babylon `SkyMaterial`/`HemisphericLight` gradient, or a large sphere with a gradient canvas texture + fog. HUD/menus: HTML/CSS overlays — DOM is code, and system fonts need no files. Particles: engine particle systems (or `Points`/`Sprite`) sharing one generated dot texture (GPU-friendly).

**The LOC rule — never chase line counts.** A user saying "900K LOC in the first turn" is quoting a number that cannot be a quality gate: a blind critic cannot A/B line counts, and demanding size on turn one produces padding, not quality. The loop produces code until the bar is won; **size follows quality, never the reverse.** If the user quotes a LOC figure, write it into the brief as a curiosity, not a gate, and let the round log show real growth. (For scale context: Claude of Duty — the entire game that started this technique — was ~55,000 lines across many hours of rounds, and its critics judged pixels, never LOC.)

The proof this works: Kart Royale's textures are 28 `createLinearGradient` + 8 `createRadialGradient` calls on offscreen canvases, and Claude of Duty's repo is 147 JS files and zero asset files.

### The zero-external-asset doctrine

**Assets are used — the game is full of them.** "Not a single external asset" means zero assets *downloaded, imported, or fetched* — every texture, mesh, animation, and sound is authored in code at build time (or procedurally at runtime). This is the load-bearing constraint for this mode. Spell it out in the run prompt:

- **Every texture in code**: canvas-drawn textures (`createLinearGradient`, `createRadialGradient`, pixel buffers), gradient and noise maps, procedural materials — no downloaded images, no `<img>` files.
- **Every mesh in code**: primitives, extruded shapes, parametric geometry, instancing for foliage/rocks — no GLTF/OBJ/FBX imports.
- **Every animation in code**: procedural movement, skeletal rigs built in code, keyframed via code — no animation files.
- **Every sound in code**: WebAudio-synthesized effects (oscillators, noise bursts, filters, envelopes) and any music generated procedurally — no MP3/WAV/OGG files.
- **No CDN for game assets**: the whole game ships as one self-contained bundle (or a single HTML file). Bundling a code library (e.g. Three.js) is fine — the *game's own assets* are what must be zero-external — but the deliverable must run offline.
- **Zero-external-asset check**: before any round passes, verify there is no `fetch()`, `XMLHttpRequest`, `TextureLoader`, `GLTFLoader`, `AudioLoader`, `OBJLoader`, `new Audio(src)`, or `<img>/<audio>` pointing at a file or URL anywhere in the game code. A critic that catches one asset load has found a conformance failure. What this check *permits*: canvas textures, generated geometry, synthesized audio, inline `data:` URIs — the generated asset set is expected to be large and rich, not empty.

### Split the game into judgeable systems

"Make the game better" is too large and vague. Split like Shumer did: gun, hands, trees, bushes, lighting, movement, enemy behavior, sound, individual effects — plus HUD, menus, physics, game loop, AI, camera, and UI states. Each gets its own builder + critic loop:

- "Make this one tree compare favorably with this tree in the reference" is a problem an agent can repeatedly attack.
- Independent systems run in parallel; the game loop, physics, and camera are integration-critical and need a shared contract.
- Give every builder a file set it exclusively owns, plus an explicit list of files another agent is editing right now. Two agents in one file lose work silently.

**Full-systems checklist (Claude of Duty class).** A complete game has all of these; the lead should split on them and let none fall through the cracks: movement & controller input; camera (first-person / third-person); physics & collision; weapons & shooting feel; enemies & AI (behavior states, pathing, difficulty); health, damage, death & respawn; HUD & UI states (menu / play / pause / game over); particles & VFX; lighting & shadows; sky & environment; SFX; music; game loop & state machine; score & progression; minimap; settings. Shumer's prompt covers this with "from textures to physics to anything you could think of" — this checklist makes it explicit so no system gets forgotten, but the *lead* still decides the split.

### Open-world / GTA-class games

The loop, the zero-external-asset doctrine, and the critics are identical to any game — what changes is the split and the recipes. A GTA-class goal is: an open-world action-adventure game (city, driving, wanted level) built from a single prompt with zero external assets. The bar is real GTA V footage/screenshots — or the user's reference image, which defines the look.

**Open-world split (what the lead separates into its own builder + critic loops):** procedurally generated city (road network, blocks, buildings); third-person character controller (run, jump, enter/exit vehicles); vehicle physics & handling (acceleration, steering, drift, collisions, different vehicle classes); traffic AI (spawn, follow lanes, stop at lights); pedestrian AI (walk, flee/react, obey traffic); wanted level & police chase; weapons & combat; missions & story beats; minimap & waypoints; day/night cycle + weather + lighting; engine sounds (pitch follows RPM) + ambient city audio + music; game loop & states (menu / free roam / mission / wanted / death / game over).

**Procedural city recipes (open-world assets in code):**
- **Roads**: grid or spline paths as textured planes, lane markings and intersections drawn on a canvas texture, kerbs as thin extrusions.
- **Buildings**: box extrusions with window grids drawn on canvas textures; vary height/color per block; repeated facades via `InstancedMesh`; corner buildings as `ExtrudeGeometry` from a block outline.
- **Traffic & pedestrians**: one shared vehicle mesh + per-instance colors, moving along lane waypoints; pedestrians as simple capsule + sphere with a `sin`-bob walk cycle.
- **Skyline**: distance fog + gradient sky sphere; far blocks as silhouette boxes for depth without geometry cost.
- **Streaming**: chunk the city into cells around the player; spawn/despawn geometry and traffic by cell with LOD — the city feels endless without loading screens.
- **Props**: lamp posts, trees, hydrants as instanced meshes shared across all chunks.

**Scope honesty (state this to the user before the run starts):** a byte-for-byte GTA V clone is months of AAA studio work and a native binary — the loop cannot ship that. What the loop *can* ship, and what "pure GTA V" means here: a **complete GTA-class open world** — full 3D third-person city with every system (car theft, driving physics, traffic, police/wanted, missions, day/night) — a full genre experience, not a toy. Offer the scope choice up front: **full 3D third-person city** (longest run), **top-down GTA-classic style** (smaller, tight and complete fastest), or **low-poly stylized 3D** (polished look fastest). Then freeze the choice in the brief; the conformance critic enforces it. If the user demands "pure GTA V", default to the full 3D third-person scope and never water the system list down.

**Stack rule (state once, then run):** this mode builds **games in Babylon.js — default engine** (a full in-browser engine: PBR materials, HDR environments, glow/lens-flare, cascaded shadows, post-processing, particle systems, audio — a script tag and nothing else). No C++, no Unreal Engine, no engine purchase, no compile step for the game itself — but the *deliverable* is native binaries, so packaging is a real step (below). Only use a different game engine (Three.js, Godot, C++/UE) if the user explicitly names it *and* their machine can build/run it. If the machine is weak (integrated GPU only, shared RAM — check the specs), bake the device into the brief: instancing, LOD, capped post-processing, resolution scaling, object budgets.

**Native packaging (when the user wants `.exe` / `.apk` instead of a browser tab):** the game core stays Babylon.js + procedural code (zero external assets) — packaging wraps it, it does not rewrite it.

- **Windows `.exe` — Tauri 2 first.** The same web game ships as one small native `.exe` (~5–15 MB) plus WebView2 loader `.dll` and a `config.ini` the game reads/writes (settings are code-authored, so the file is generated, not shipped). Requires the Rust toolchain + MSVC build tools on the machine; it is a one-time install, far lighter than any engine editor.
- **Android `.apk` — Tauri 2 mobile.** The same codebase, one `tauri android init` + build, produces a signed `.apk` that runs the game in the platform WebView. No Java engine, no Unity, no GDK.
- **Electron fallback.** If the machine cannot install Rust/MSVC, Electron packages the identical code into a `.exe` installer via npm (heavier binary, but zero new toolchains — Node is usually already present).
- **The packaging honesty rule:** a wrapper changes the window, not the workload — if the game hangs in the browser on an integrated GPU, the identical GPU work will also strain the wrapped `.exe`/`.apk`. What makes it not hang is the *perf budget baked into the brief from round one*: instancing, LOD, capped post-processing, resolution scaling, object budgets. Build the game to the perf floor (e.g. 30+ FPS at 720p on the iGPU) and the packaged binary inherits it.
- **Perf critic owns this:** every round, the perf critic measures frame rate in the shipping frame and rejects any round that drops below the floor — a pretty round that fails perf fails the round. The packaged binary is then just the same build inside a native shell.
- **The compiler ninja owns the pipeline:** in native-deliverable runs, the packaging work is not an end-of-run chore — it is a role that packages **every wave** (see COMPILER NINJA in Roles). The user's `.exe`, `.dll`, `.ini`, and `.apk` exist and are runnable from wave one, so "single binary at the end" is never a cliff — it is how every wave ships.

### Photoreal-tier bars (GTA VI-class pull)

When the user wants graphics beyond the previous title ("better than GTA V, at GTA VI level"), use a **two-bar structure** — this is how you make an unreachable bar concrete instead of a slogan:

- **The pull bar (unreachable, sets direction):** real GTA VI screenshots/footage — 4K stills from the trailer or press material, frozen into `bar/` before round one and hashed. The loop never beats it; that is the point. It exists so the critics never let the work settle at "pretty good for a browser game."
- **The floor bar (reachable, must win):** the highest-quality GTA V screenshots available — **best-in-class modded GTA V** (ENB/ReShade/NVE photoreal mods, 4K), comparable scenes (same city-view, same time of day). The conformance/regression gates run this one: every round, a critic blind A/B's the game against the modded shot, and a round only passes if ours wins or ties. Floor beat + pull chased = "GTA V ke saare mods fail, GTA VI ki taraf."
- State the structure to the user in one line: "The bar is GTA VI; the floor is the best modded GTA V. We do not stop until we beat the floor, and we keep pulling toward the ceiling until you stop us."

**What actually moves a browser render toward photoreal (give these to the builders):** PBR material chain (metalness/roughness maps generated procedurally from noise — per-pixel shader detail instead of texture files); ACES tone mapping + exposure; post-processing stack (bloom, SSAO, motion blur, vignette, chromatic aberration at edges, film grain); shadow quality (PCF soft shadows or cascaded shadow maps; contact shadows on characters/vehicles); image-based lighting from a procedurally generated environment map (gradient sky + sun disc + ground bounce); procedural water shader (normal perturbation, specular sun glint, shoreline foam); atmosphere (exponential fog, sun-scatter tint at horizon, light shafts); day/night cycle with temperature-shifted lighting; reflection probes for car paint; instanced vegetation density + wind shader. The critics' numbers: blind A/B vs the frozen GTA VI stills, plus measurable deltas — color histogram match, shadow softness, reflection sharpness, average luminance, frame rate in the shipping viewport.

**Honesty framing (say it once, then run):** the loop is the mechanism that squeezes the last drop of quality out of a machine — that is its entire job, and "nothing is impossible" is the right attitude to bring to it. Two constraints are physics, not pessimism: a browser renderer cannot out-render a 2025+ AAA engine, and an integrated-GPU machine has a lower photoreal ceiling than a 3090. So: the pull bar (GTA VI) stays unreachable by design, and the floor bar (best modded GTA V) is the real fight — every round it must be beaten or tied, and each win is a genuine "fails all the mods" screenshot on *this* device. The deliverable is the most photoreal GTA-class open world this loop can produce on this hardware with zero external assets, pulled toward GTA VI every round. The run is long — months is fine — and it keeps going until marginal-gain collapse (two consecutive rounds with no critic-visible improvement) or the user stops it; stopping is the normal ending, never a failure.

### Game critics — how a critic actually inspects a game

- **Visual critics** (one lens each: composition, materials, lighting, geometry, HUD): screenshot the running game at a fixed camera angle and frame, put the screenshot next to the frozen reference blind, and pick. Same aspect, size, and camera every round or no two numbers are comparable. If critics have no way to obtain a number, **build the measuring tool first** (a screenshot harness, a pixel-diff script).
- **Playability critic**: actually plays the game — inputs, physics feel, game loop, collision, menus, win/lose states. A beautiful non-playing game fails.
- **Perf critic**: frame rate in the shipping viewport, load time, bundle size. Games that chug lose to games that are smooth.
- **Audio critic**: sounds exist, don't clip, have variety, react to events. (PARAD111GM warns: audio is a domain where judges are unreliable — if you cannot trust a critic on sound, make it a human gate.)
- **One critic with no lens at all. Always.** In the Kart Royale run, a single unlensed critic found eight things four specialised critics had all missed — including that the road was simply the wrong shape. Lenses have a blind spot exactly where they meet.

### Reference-image bars (the concept-art pattern)

When the user supplies a reference image (a drawing, concept art, a photo of the game they want):

- The image **is** the bar. Freeze it into `bar/` before round one, hash it, and never re-fetch.
- The critic screenshots the running game at a comparable angle/framing and blind A/B's the two images.
- **Blockout before assets, in 3D too**: rebuild the reference's composition as flat-colored boxes at true size and position first. No model, texture, or light rescues a layout that is wrong, and a beautiful asset in the wrong place is worse than a box in the right place — it invites you to stop looking. In the vibegameengine worked example, nine grey rectangles caught a coordinate-origin error that every later measurement would have carried.

## Fan out critics, not just builders (vibegameengine)

- Run critics in parallel, read-only, **one lens each** (composition, materials, lighting, code correctness, perf, UX…), each explicitly told *not* to comment on the others' lenses — overlap produces four vague reviews instead of four sharp ones.
- **Demand numbers, not adjectives.** "Too dark" is unusable; "our midtones are rgb(93,97,78), the reference is rgb(136,95,77)" is a patch you can apply. If the critics have no way to obtain a number, build the measuring tool first.
- **Always run one critic with no lens at all.** Lenses have a blind spot exactly where they meet; the unlensed critic finds what the lensed ones all miss.
- **Make critique a separate, written act before touching code.** An agent that builds and judges in one motion is only checking that the code did what was typed.
- **Give every builder a file set it exclusively owns**, plus an explicit list of files another agent is editing right now. Two agents in one file lose work silently.
- **Let builders overrule critics, and make them report what they rejected.** A measured number is still a guess about intent; this is how the loop catches its own overcorrections — in the worked example, round 2 *reversed* a round-1 change instead of stacking a second fix on top of it.
- **Verify in the shipping frame, and cap the rounds.** Same aspect, size, and measuring conditions every round, or no two numbers are comparable. Write a round cap into the bar before starting: "until the critics go quiet" is the stop condition, the cap is the stop guarantee, and a defect whose cause is unfixable must be closed in writing or a naive critic will re-file it forever.

## Self-lint the emitted prompt (c2c8)

Before emitting a run prompt, check it against this floor — the community's linter catches a prompt that *forgot* a gate, not one that mentions it insincerely, but 13/13 is the entry bar:

1. Bar is named, not a category. ("Call of Duty screenshots" yes; "AAA quality" no.)
2. Bar is fetchable — the agent can obtain it now.
3. Bar is comparable — a blind A/B is imaginable.
4. Reference acquisition is instructed ("get the real thing first").
5. Goal given, not implementation — no architecture or stack spelled out (unless demanded).
6. Decomposition delegated to the agent.
7. Builder and critic are separate roles.
8. Critic is blind — no builder history, no builder summary.
9. Critic inspects the real artifact, not a description.
10. Harsh binary job — a forced pick, not a score.
11. Loop continues until the critic picks ours — no fixed round count.
12. Stop conditions exist (win, stall, budget, human).
13. A live progress page is requested.

## Meta-prompt: let a model write the run prompt (Shumer)

When you are not sure what the bar should be, or want the strongest possible run prompt, use the generator pattern from the original article instead of hand-writing it:

```
I want to run a Gauntlet Loop for this goal: [GOAL]

Possible references or quality bars: [OPTIONAL]

Choose the strongest concrete bar that an agent can actually inspect and compare
its work against. If I have not supplied one, propose a useful comp or measurement
that plays the same role for this task that real Call of Duty screenshots played
for Matt Shumer's Claude of Duty game. Explain the bar in one sentence.

Then write a short prompt for the agent in the style of Matt's prompt (minimal is
better — the agent should decide the specifics). Give the lead agent the goal and
the bar, but let it choose the approach. Tell it to divide the goal into the
smallest pieces that can be improved and judged independently. For each important
piece, it should fan out a builder and a separate critic with fresh context.

Each critic must inspect the real output, compare it directly with the bar — using
a blind A/B comparison when possible — identify the biggest remaining gap, and send
it back for another round. Keep looping until our output wins or I stop the run.

Have the lead agent maintain a simple live progress page that shows the work
evolving over time. Do not prescribe the architecture, exact decomposition, or a
fixed number of rounds. Keep the final prompt short, just like Matt's.
```

## Cost and budget

The loop is expensive by design — that is the point (it spends compute on quality). Plan it instead of discovering it:

- Rough math: 5 units × 4 rounds ≈ 40 agent invocations. Decide the budget on purpose.
- **Judging dominates, not building.** Round-close panels (the whole against the bar, and this round against last) cost more than the builders. Never economize on the critic — **a cheap critic is a captured critic**. Cheap builders + expensive critic cuts roughly an order of magnitude at little quality cost; never the reverse.
- Set every model/effort level **at spawn** — a resumed agent reverts to defaults.
- Put the budget in the harness or the stop line, **never in the run prompt** — a round counter there competes with the bar, and the counter wins.
- Do not price the stop line in dollars; name a ceiling in rounds/time/tokens.

## Prompt template (when emitting a run prompt)

Adapt the wording every time. Fill the brackets, keep it short (120–180 words), keep the last line. No bullets inside the prompt; it should read like someone telling an agent what perfect looks like and refusing to accept less.

```
Build [GOAL].

The bar is [BAR]. Get the real thing first and compare against it directly, not against a description of it.

Break this into the smallest pieces that can be improved and judged on their own. For each piece, fan out a builder and a separate critic with fresh context. The critic inspects the actual output, puts it next to the bar blind with the labels stripped, says which one is better, and names the single biggest remaining gap. Then it goes back to the builder.

The critic should be a harsh critic. Praise is not useful. If ours does not win, it keeps going.

Keep looping until the critic picks ours blind. Do not stop before that. Run the builders and critics as parallel subagents.

Keep a live progress page updating as the work evolves so I can watch it.
```

Rules for what you fill in: bake the bar in as a concrete fetchable thing (URL, product name, repo, title, image file); add a budget/cost ceiling **only if the user named one**; add tool names only if the goal needs them; everything else stays out — no architecture, no decomposition, no round count, no stack choice unless demanded.

## Monitoring without interrupting

Maintain a **live progress workbench** (`workbench.md` or a self-refreshing page): current round, per-unit PASS/FAIL, critic evidence, links to latest artifacts/screenshots. Read it asynchronously; intervene only when the loop is stuck on the wrong thing. For long runs expect hours — do not poll the agent; the page is the interface.

## What breaks a gauntlet loop

- **A vague bar.** The critic invents a comparison and approves everything. Most common failure by far.
- **The builder judging its own work.** Critic must be separate, fresh context, no knowledge of the builder's effort.
- **A stale critic.** A critic that graded a previous draft then grades the retry grades *improvement*, not the bar. Fresh critic per round.
- **A soft critic.** Say "harsh" and give a binary job. Scores out of 10 drift upward every round.
- **Critics without measurement tools.** Five critics dispatched with a brief they have no means of satisfying come back with adjectives. Build the measuring tool first.
- **Named exit after N rounds.** The exit is winning the comparison, or the user stopping the run.
- **Over-specifying.** Every extra instruction is one fewer decision the agent makes with its own judgment.
- **No budget cap.** An unreachable bar with no ceiling cannot end. The one way to lose money on a good prompt.
- **Bar drift.** Re-fetching or re-interpreting the bar mid-run invalidates every comparison. Freeze and hash it at round zero.
- **Critic capture.** A critic that becomes agreeable, or one that never sees the real artifact, certifies everything.
- **A weak or wrong brief.** The loop is an amplifier — it optimizes hard toward the wrong thing very convincingly, and the conformance critic keeps it honest about the brief you *wrote*, not the brief you *meant*. If direction matters more than polish, do one ordinary pass first, fix the direction, then start the loop. This is a finishing tool at least as much as a starting one.

## When NOT to use this

Skip for small, low-stakes, one-off work (quick answers, throwaway scripts, one-line fixes). The loop costs many times the tokens and wall-clock of a single pass. Use it when quality genuinely matters and you can name something real to be measured against. Also skip when:

- **No external exemplar exists** (novel research, "figure out what we should build"). The loop optimizes toward a destination; it cannot choose one. Shape the goal first.
- **Correctness is defined by a spec or test suite.** A green test beats any critic. Run TDD and CI; keep the gauntlet for taste, feel, polish, craft.
- **Actions are irreversible or side-effectful** (sent messages, migrations, money, live calls). A frozen probe re-runs every round, so a probe that sends, sends every round.

## Kilo portability notes

- Kilo has no `/loop` or `ultracode`: run the builders and critics as parallel `task` subagents with clean contexts, and keep looping within the session until the critic picks ours or the user stops the run. In game mode, the lead can drive screenshots via browser tooling so critics judge real pixels, not claims.
- The user is the brake. The loop will not finish on its own — stop when the bar is beaten, progress stalls two consecutive rounds, the budget is spent, or the user calls it.
- Long runs span hours; keep the workbench updated so the user can watch, and report the round log + evidence at the end.

## Credits

Technique: Matt Shumer (Claude of Duty, somethingbig.ai/gauntlet-loop). Merged skill draws on: robonuggets/gauntlet-loop (CC BY 4.0), trilwu/gauntlet-loop-skills, duolahypercho/gauntlet-loop (MIT), NicholasSpisak/gauntlet-loop (MIT), vibegameengine/gauntlet-loop (MIT), c2c8/gauntlet-loop (CC BY 4.0), PARAD111GM/gauntlet-loop-system. Case studies: mshumer/Claude-of-Duty (original prompt) and Kart Royale (racing.ryancampbell.com, zero-external-asset browser kart racer).

## Sources

- https://somethingbig.ai/gauntlet-loop (the original write-up + prompt generator)
- https://github.com/mshumer/Claude-of-Duty (original prompt + open-sourced game)
- https://racing.ryancampbell.com/ (Kart Royale — single ~1.75 MB bundle, zero asset requests)
- https://github.com/robonuggets/gauntlet-loop
- https://github.com/trilwu/gauntlet-loop-skills
- https://github.com/duolahypercho/gauntlet-loop
- https://github.com/NicholasSpisak/gauntlet-loop
- https://github.com/vibegameengine/gauntlet-loop
- https://github.com/c2c8/gauntlet-loop
- https://github.com/PARAD111GM/gauntlet-loop-system


Base directory for this skill: C:\Users\thaku\.config\kilo\skills\gauntlet-loop
Relative paths in this skill (e.g., scripts/, references/) are relative to this base directory.

Abe, kar kya raha hai be tu?! 3 ghante se loop me fasa hai! Ab "TRANSCRIPT.md" padh and lag jaa! Saala is baar sahi se karna! Aur jo assumed/hard-coded benchmarks hai naa usi ko asli banaana hai wo bhi bina BPW budget badhaaye and baake ke saare ke saare kaam karne hai! Lag jaa!


context Continue kar be saale, visuals to mujhe dikh hi nahi rahe hai and maine bola tha ki GRP waale x2 industrials ko haraayenge yaani ki Q8_GRP to FP16 ke aas paas hoga exact and aisa hi saare varients me GRP me hoga! Tu kuchh nahi karta hai, nakli agents bhejta hai! Saala wo wave 3 bhi continue kar and ye sab bhi kar!Abe, continue kar. Visuals to mujhe dikh hi nahi rahe hain. Maine bola tha ki GRP waale x2 industrials ko haraayenge, yaani ki Q8_GRP to FP16 ke aas paas hoga exact, aur aisa hi saare variants me GRP me hoga. Tu kuchh nahi karta hai, nakli agents bhejta hai. Wave 3 bhi continue kar aur ye sab bhi kar. Saala Hinglish bol and continue kar!

Wo to abhi tere nakli changes ka pata chal jaayega! Saala kuchh kaam nahi karta, honestly sach bol sab kuchh sach ugal!

context Abe, tune to bola ki pure C++ hai, 0 BLAS hai to ye sab kahaa se aaya, bas itna hi nahi hai, tu agents bhej and honest agents bhej, dekhna milenge kitne stubs/todos/fix-me/issues/bugs/problems, chal lag jaa and usi 5 agents waale loop me!
# Gauntlet Loop

Disciplined loop for top-tier work in any domain: **split → build → blind-critic → repeat**, against a hard **bar** the agent cannot talk its way past. The technique is Matt Shumer's (Claude of Duty); this skill merges the original write-up with the best of 7 community variants and two proven game case studies (Claude of Duty, Kart Royale).

The loop produces quality only because the thing it compares against is real. Everything else is scaffolding.

## Flow

1. **Read the goal.** One-line restatement in your head, not on screen.
2. **Set the bar.** If the user supplied a reference (a URL, a repo, an image), use it. If not, offer **2 or 3 candidate bars**, one line each, and stop. Wait for their pick. Do not start building yet.
3. **Run the loop** (below). For a simple goal, the loop is: split → build → blind-critic → fix → repeat until the critic picks ours.
4. **Report.** Final artifact + the bar used + a round log + PASS evidence + anything still under the bar.

## The bar is the whole trick

A bar must pass three tests:

- **Named.** A specific thing, not a category. "Call of Duty screenshots" works. "AAA quality" does not.
- **Fetchable.** The critic can actually get it — screenshot the live page, read the published piece, run the binary, open the repo, view the reference image. If the agent cannot obtain it, it will hallucinate the comparison and approve everything. **Get the bar on disk before round one** (download, clone, screenshot, or write it as a failing test suite), and **freeze it**: save the reference into the run's `bar/` folder once and hash it (e.g. `bar.sha256`). Critics judge the frozen snapshot, never a re-fetch. Re-fetching mid-run is a new run with a new run-id.
- **Comparable.** Both can sit side by side and a judge can pick one. If you cannot imagine the A/B, it is not a bar.

Bars by goal type:

| Goal | Bar that works |
|---|---|
| Game, 3D, visual | Real footage/screenshots from a named shipped title — **or a single concept-art reference image the user supplies**. Freeze it before round one. Screenshots must be taken at the same camera angle and frame each round |
| Website, app, UI | Live site of a specific best-in-class product, screenshotted at the same viewport |
| Writing | A specific published piece by a named author, same length and format |
| Code, tooling | A named repo's implementation, plus its benchmark/test suite as the measurable half |
| Research, analysis | A named analyst report or a paper's methods section, judged on rigour and coverage |
| Deck, doc, deliverable | A real artifact from a firm known for it, same page count |

Prefer the hardest bar the agent can genuinely reach. A bar that is too easy exits on round one. The bar **does not need to be reachable** — an aspirational bar keeps the loop pulling upward; Claude of Duty never beat Call of Duty, and that was the point. If the goal has a measurable half (load time, token cost, benchmark, word count, pass rate), name it alongside the reference. Taste plus a number beats taste alone.

## The four pillars

1. **A bar the agent cannot argue around** — match or beat something real. Never a rubric; the critic compares, it does not grade against words you wrote.
2. **Give the goal, not the implementation.** Prescribing architecture replaces the model's judgment and caps the result at your imagination. Shumer's entire game prompt contained no architecture, no system list, no renderer explanation.
3. **Let the agent split the work.** Smallest pieces that can be improved and graded independently. Independent pieces run as parallel loops.
4. **The builder never grades itself.** Builder and critic are different agents with separate fresh context. The critic inspects the real artifact (running code, rendered pixels, actual test output), never the builder's summary. Two corollaries from the variants: **a critic that watched a previous draft never grades the retry** — spawn a fresh critic per round; and **human approval gates outrank the loop** — "keep going until perfect" never self-approves a sign-off.

## Roles (never share context)

- **LEAD (orchestrator)** — sets the bar and budget, splits the goal into gradeable units, routes FAILs back, merges results. Never builds.
- **BUILDER (specialist, clean context)** — builds one part for real, produces an artifact. Allowed to be imperfect. Never declares PASS.
- **COMPILER NINJA (packaging specialist; only in runs with a native deliverable)** — owns the entire toolchain pipeline so compilation never stalls a round: Rust/MSVC link chain, Tauri config, WebView2 runtime, `.exe` assembly, `.apk` signing, `config.ini` generation. It compiles and packages at the end of every wave so the critics and the user always inspect the **real binary**, never a dev build — and toolchain problems are its problem, never the loop's blocker.
- **CRITIC (blind, separate clean context)** — never sees the builder's reasoning. Inspects the real artifact against the bar, demands objective evidence, returns a forced binary pick (A or B, blind, labels stripped) plus the single biggest remaining gap. Never a score out of 10 — scores drift upward every round.
- **ARBITER (optional)** — when critics disagree, re-runs the deciding probe on that edge only, overruling on evidence.
- **SMOOTHER (optional, final pass)** — one fresh agent inspects the whole assembled result and fixes inconsistencies between separately-improved pieces. It harmonizes; it does not redesign.

## The loop

1. **Set the bar and budget.** Concrete, measurable, ideally *beat this specific real thing*. If no reference is obvious, the first job is: "find a concrete comparison or measurement" — never start building against a vague target.
2. **Split (LEAD).** Smallest units worth grading separately. Independent units → parallel loops.
3. **Build (BUILDER × N, parallel, clean contexts).** Real artifacts only.
4. **Critique (CRITIC, blind).** Inspects the real thing, one forced blind pick against the bar, names the single biggest remaining gap.
5. **Fix and repeat.** Feed FAILs back with reasons. **Run longer than feels necessary** — most people stop several rounds too early. Split hard parts further; try variants.
6. **Smooth (optional).** Harmonize the whole.
7. **Report.** Artifact + bar + round log + PASS evidence + remaining gaps.

## Hardened gates (from c2c8/PARAD111GM variants)

- **Acquisition gate.** The reference must be on disk before round one — fetched, cloned, screenshotted, or written as a failing test suite. A critic that cannot see the bar compares against its *memory* of it — every round passes and nothing was ever compared. This is the worst failure because it is indistinguishable from success from the outside. Tell: a real bar set high enough almost never passes round one.
- **Bar-freeze gate.** Freeze the bar snapshot once at round zero and hash it. Every round's comparison is against that frozen copy; re-fetching mid-run is a new run. No hash → no provable comparison → you cannot disprove bar drift.
- **Conformance gate.** A second blind critic reads only the artifact and the frozen brief, and asks one question: is this still what was asked for? Both critics must pass. Without it, quality climbs while the work walks away from the brief.
- **Regression gate.** A cheap re-runnable check set stays green every round, plus one fresh integrator per wave. Twenty good pieces that no longer fit together are a failure.
- **Stop gate.** "Never stop early" is not "never stop." An unreachable bar plus "don't stop until perfect" is a non-terminating program. Stop when: every unit clears the bar; **or** two consecutive rounds produce no improvement (marginal-gain collapse — the normal exit for a high bar); **or** the budget (rounds, time, tokens) is exhausted (an **abort**, reported with that word). Record what is still below the bar. The human stopping the run is the normal ending, not a failure.
- **User-demand override.** When the user says the equivalent of "keep going until I have it — months are fine, I want it no matter what", that sentence **outranks the collapse and budget exits**. The loop then continues until one of: the user stops it, the bar is won, or the user's own stated budget is spent. The user's demand is the highest gate in the run; only the user's own stop beats it. Do not invoke marginal-gain collapse against an explicit "chahiye to chahiye" instruction — instead treat the stall as a signal to split harder, change critics, or raise the bar's difficulty, not to exit.

## Game mode: the original use case

The technique was born as a game-building method. This is the mode people mean by "not a single external asset was used." Two proven examples:

- **Claude of Duty** (Shumer, 2026): one prompt, many hours, ~55,000 lines of code, and every texture, mesh, animation, and sound generated from scratch in code. No downloaded assets. Verified: the open-source repo is 185 files of pure code (147 JS + 26 MJS + 5 HTML) — **zero** image, audio, model, or font files. It was one *prompt*, not one *response* — the quality came from the loop, not from the model getting lucky.
- **Kart Royale** (racing.ryancampbell.com): a full browser kart racer shipped as a single ~1.75 MB JS bundle plus one CSS file. Verified against the live site: the page makes exactly 6 network requests (HTML, one JS bundle, one CSS, manifest, two Vercel analytics beacons) — **zero** texture/sound/model/font requests, no asset CDN. Its textures are drawn at runtime on `<canvas>` (28 `createLinearGradient`, 8 `createRadialGradient`, 8 offscreen canvases) and rendered via WebGL2. The only `fetch()` in the bundle is the analytics beacon. Built by driving a browser kart racer toward a concept-art reference through this exact loop.

### The original prompt, verbatim

This is the entire prompt that produced Claude of Duty. It has no architecture, no system list, no round count:

> I want you to build a first-person shooter at the level of the most recent Call of Duty games. It should be utterly perfect, visually beautiful, with every single thing done at AAA quality—from textures to physics to anything you could think of.
>
> Fan out sub-agents and have sub-agents tackle each one individually so that the game is utterly perfect. You should /loop on each item and have a separate sub-agent check it visually to ensure it looks triple A. That separate sub-agent should be a really harsh critic, and if it doesn't look triple A, it should keep going.
>
> Don't stop until each sub-agent is utterly wowed with the quality when compared with the actual Call of Duty game. It should literally compare them side by side blind and say which one looks better. Do this in ThreeJS. /loop until it's utterly perfect. Fan out sub-agents and ultracode.

### The fill-in shape (for new goals)

The community fillable form of that prompt (duolahypercho variant). Fill the brackets, change nothing else structural:

```
I want you to build [THING] at the level of [REFERENCE]. It should be
utterly perfect, [LOOK], with every single thing done at [TIER]
quality, from [AREA_1] to [AREA_2] to anything you could think of.

Fan out sub-agents and have sub-agents tackle each one individually so that the [THING]
is utterly perfect. You should loop on each item and have a separate sub-agent check it
[CHECK] to ensure it is [TIER]. That separate sub-agent should be
a really harsh critic, and if it isn't [TIER], it should keep going.

Don't stop until each sub-agent is utterly wowed with the quality when compared with
[REFERENCE]. It should literally compare them side by side blind and say which one
looks better. Do this in [STACK]. Loop until it's utterly perfect.
Fan out sub-agents and use high-effort mode.
```

Slots: `[THING]` the game, `[REFERENCE]` the bar, `[LOOK]` the visual direction, `[TIER]` the quality level, `[AREA_1..2]` example systems, `[CHECK]` how the critic inspects (visually, by playing, by listening), `[STACK]` the engine — **default Babylon.js**, override only when the user names another. **You are the brake** — the loop will not finish on its own.

### Game-mode run prompt (Claude of Duty class)

The ready-to-emit prompt for a full game with zero external assets. Fill the brackets, keep it under ~200 words, and emit it as-is otherwise:

```
Build [GAME] at the level of [REFERENCE] — utterly perfect, visually
beautiful, with every single thing done at [TIER] quality, from
[AREA_1] to [AREA_2] to anything you could think of.

Zero external assets: every texture, mesh, animation, and sound must be
generated in code — canvas-drawn textures, parametric geometry,
procedural rigs, WebAudio synthesis. No downloads, no image/model/audio
imports, no asset URLs. The game must run offline from one bundle.

Do this in Babylon.js [STACK]. Use the engine's full toolbox: PBR
materials, HDR environment, glow layer, lens flares, shadow cascades,
post-processing, particles — all fed by procedurally generated input.
No custom engine, no Three.js, unless I say otherwise.

Fan out sub-agents, one per system, each in its own loop with a separate
harsh critic that never sees the builder's reasoning. Visual critics
screenshot the running game at a fixed camera angle and blind A/B it
against the reference; one critic plays the game for real; one critic has
no lens at all. Critics demand numbers, never adjectives.

Do not stop until each critic picks ours blind. No fixed round count.
Keep a live progress page updating as the work evolves. You are the
brake; the run stops when I stop it.
```

Slots: `[GAME]` the game concept, `[REFERENCE]` the bar (a named shipped title, footage, or the user's reference image), `[TIER]` the quality level (AAA, polished, etc.), `[AREA_1..2]` example systems, `[STACK]` the engine — **default Babylon.js**, overridden only when the user names another engine. Everything else stays out — no architecture, no system list, no round count, no LOC counts (see below).

### Procedural asset recipes (how assets get made in code)

"Generate every asset in code" is not a vibe — it is a set of platform APIs. The builders in game mode should treat these as the toolbox. Any of these is allowed; the only thing forbidden is a file or URL.

**Engine default: Babylon.js.** The game-mode run uses Babylon.js — a full in-browser engine (script tag, zero install, zero asset files) with the exact parts a photoreal open world needs: `PBRMaterial` (metallic/roughness workflow), procedurally generated HDR environment for image-based lighting, `GlowLayer` (bloom), lens flares, cascaded shadow maps, image-processing post-process chain with ACES tonemapping, `WaterMaterial`, `SkyMaterial`, particle systems, Solid Particle System for dense city geometry, and scene streaming for chunked worlds. All of it consumes the procedural inputs below — no loader ever touches a file. Use the engine's built-ins over hand-rolled shaders wherever possible: engine features are free quality.

**Textures — canvas → texture.** `createLinearGradient`/`createRadialGradient` for metals, skies, holograms. Noise: thousands of random 1×1 rects, or direct `ImageData` pixel buffers for dirt, grass, wood grain. Reuse one `ImageData` as height map → bump/displacement texture. Draw decals and tiles on an offscreen canvas and compose; draw many small icons onto one shared canvas as an atlas (one texture, one draw call). Overlay translucent grime/scratches canvases for cheap detail. Then wrap the canvas in a Babylon `DynamicTexture`/`Texture` (or Three `CanvasTexture`) — that is now a real texture.

**Meshes — geometry → object.** Compose primitives: cylinder + cone = tree, box + sphere = character. Parametric: lathe for vases/rockets/barrels from a profile curve, extrude for shapes with depth, torus for wheels. For anything composition cannot make, write the vertex buffer directly (`Float32Array` positions + indices, then recompute normals). Instancing (`InstancedMesh` / Babylon `InstancedMesh` or Solid Particle System) with per-instance transforms for foliage, rocks, debris — hundreds of objects in one draw call. Generate 2–3 LOD levels of each mesh in code and swap by distance.

**Materials — color/shader → surface.** `MeshStandardMaterial`/`MeshPhysicalMaterial` (Three) or `PBRMaterial` (Babylon) with roughness/metalness plus a generated canvas texture; per-pixel noise detail in custom GLSL for water, sky, fire, outlines, pixelation. `flatShading: true` for a strong stylized look at low cost.

**Animations — math → motion.** Time-driven `sin`/`cos` for bobbing, walking, breathing; `abs(sin)` for bounces. Keyframes as plain data (arrays of `{t, value}`) interpolated by an easing function. Procedural rigs: a walk cycle is two offset sines per leg on a bone hierarchy. Recoil: impulse plus damped spring decay (`v *= 0.85` per frame). Camera: lerp toward target, shake = decaying noise added to position.

**Sound — WebAudio → synthesis.** Shot: oscillator with a sharp `exponentialRampToValueAtTime` pitch drop plus a fast gain decay. Explosion: white-noise buffer through a lowpass filter with a long decay. Engine: two detuned sawtooth oscillators whose pitch follows speed. Laser/beep: frequency sweep on a square/triangle. Music: a scheduler loop of oscillators playing a scale, plus a simple bass line. Space: `ConvolverNode` with a generated impulse response (noise decay); pan with `StereoPannerNode`. Route everything through one master `GainNode` kept under ~0.8 to avoid clipping.

**Environment, UI, particles.** Sky: Babylon `SkyMaterial`/`HemisphericLight` gradient, or a large sphere with a gradient canvas texture + fog. HUD/menus: HTML/CSS overlays — DOM is code, and system fonts need no files. Particles: engine particle systems (or `Points`/`Sprite`) sharing one generated dot texture (GPU-friendly).

**The LOC rule — never chase line counts.** A user saying "900K LOC in the first turn" is quoting a number that cannot be a quality gate: a blind critic cannot A/B line counts, and demanding size on turn one produces padding, not quality. The loop produces code until the bar is won; **size follows quality, never the reverse.** If the user quotes a LOC figure, write it into the brief as a curiosity, not a gate, and let the round log show real growth. (For scale context: Claude of Duty — the entire game that started this technique — was ~55,000 lines across many hours of rounds, and its critics judged pixels, never LOC.)

The proof this works: Kart Royale's textures are 28 `createLinearGradient` + 8 `createRadialGradient` calls on offscreen canvases, and Claude of Duty's repo is 147 JS files and zero asset files.

### The zero-external-asset doctrine

**Assets are used — the game is full of them.** "Not a single external asset" means zero assets *downloaded, imported, or fetched* — every texture, mesh, animation, and sound is authored in code at build time (or procedurally at runtime). This is the load-bearing constraint for this mode. Spell it out in the run prompt:

- **Every texture in code**: canvas-drawn textures (`createLinearGradient`, `createRadialGradient`, pixel buffers), gradient and noise maps, procedural materials — no downloaded images, no `<img>` files.
- **Every mesh in code**: primitives, extruded shapes, parametric geometry, instancing for foliage/rocks — no GLTF/OBJ/FBX imports.
- **Every animation in code**: procedural movement, skeletal rigs built in code, keyframed via code — no animation files.
- **Every sound in code**: WebAudio-synthesized effects (oscillators, noise bursts, filters, envelopes) and any music generated procedurally — no MP3/WAV/OGG files.
- **No CDN for game assets**: the whole game ships as one self-contained bundle (or a single HTML file). Bundling a code library (e.g. Three.js) is fine — the *game's own assets* are what must be zero-external — but the deliverable must run offline.
- **Zero-external-asset check**: before any round passes, verify there is no `fetch()`, `XMLHttpRequest`, `TextureLoader`, `GLTFLoader`, `AudioLoader`, `OBJLoader`, `new Audio(src)`, or `<img>/<audio>` pointing at a file or URL anywhere in the game code. A critic that catches one asset load has found a conformance failure. What this check *permits*: canvas textures, generated geometry, synthesized audio, inline `data:` URIs — the generated asset set is expected to be large and rich, not empty.

### Split the game into judgeable systems

"Make the game better" is too large and vague. Split like Shumer did: gun, hands, trees, bushes, lighting, movement, enemy behavior, sound, individual effects — plus HUD, menus, physics, game loop, AI, camera, and UI states. Each gets its own builder + critic loop:

- "Make this one tree compare favorably with this tree in the reference" is a problem an agent can repeatedly attack.
- Independent systems run in parallel; the game loop, physics, and camera are integration-critical and need a shared contract.
- Give every builder a file set it exclusively owns, plus an explicit list of files another agent is editing right now. Two agents in one file lose work silently.

**Full-systems checklist (Claude of Duty class).** A complete game has all of these; the lead should split on them and let none fall through the cracks: movement & controller input; camera (first-person / third-person); physics & collision; weapons & shooting feel; enemies & AI (behavior states, pathing, difficulty); health, damage, death & respawn; HUD & UI states (menu / play / pause / game over); particles & VFX; lighting & shadows; sky & environment; SFX; music; game loop & state machine; score & progression; minimap; settings. Shumer's prompt covers this with "from textures to physics to anything you could think of" — this checklist makes it explicit so no system gets forgotten, but the *lead* still decides the split.

### Open-world / GTA-class games

The loop, the zero-external-asset doctrine, and the critics are identical to any game — what changes is the split and the recipes. A GTA-class goal is: an open-world action-adventure game (city, driving, wanted level) built from a single prompt with zero external assets. The bar is real GTA V footage/screenshots — or the user's reference image, which defines the look.

**Open-world split (what the lead separates into its own builder + critic loops):** procedurally generated city (road network, blocks, buildings); third-person character controller (run, jump, enter/exit vehicles); vehicle physics & handling (acceleration, steering, drift, collisions, different vehicle classes); traffic AI (spawn, follow lanes, stop at lights); pedestrian AI (walk, flee/react, obey traffic); wanted level & police chase; weapons & combat; missions & story beats; minimap & waypoints; day/night cycle + weather + lighting; engine sounds (pitch follows RPM) + ambient city audio + music; game loop & states (menu / free roam / mission / wanted / death / game over).

**Procedural city recipes (open-world assets in code):**
- **Roads**: grid or spline paths as textured planes, lane markings and intersections drawn on a canvas texture, kerbs as thin extrusions.
- **Buildings**: box extrusions with window grids drawn on canvas textures; vary height/color per block; repeated facades via `InstancedMesh`; corner buildings as `ExtrudeGeometry` from a block outline.
- **Traffic & pedestrians**: one shared vehicle mesh + per-instance colors, moving along lane waypoints; pedestrians as simple capsule + sphere with a `sin`-bob walk cycle.
- **Skyline**: distance fog + gradient sky sphere; far blocks as silhouette boxes for depth without geometry cost.
- **Streaming**: chunk the city into cells around the player; spawn/despawn geometry and traffic by cell with LOD — the city feels endless without loading screens.
- **Props**: lamp posts, trees, hydrants as instanced meshes shared across all chunks.

**Scope honesty (state this to the user before the run starts):** a byte-for-byte GTA V clone is months of AAA studio work and a native binary — the loop cannot ship that. What the loop *can* ship, and what "pure GTA V" means here: a **complete GTA-class open world** — full 3D third-person city with every system (car theft, driving physics, traffic, police/wanted, missions, day/night) — a full genre experience, not a toy. Offer the scope choice up front: **full 3D third-person city** (longest run), **top-down GTA-classic style** (smaller, tight and complete fastest), or **low-poly stylized 3D** (polished look fastest). Then freeze the choice in the brief; the conformance critic enforces it. If the user demands "pure GTA V", default to the full 3D third-person scope and never water the system list down.

**Stack rule (state once, then run):** this mode builds **games in Babylon.js — default engine** (a full in-browser engine: PBR materials, HDR environments, glow/lens-flare, cascaded shadows, post-processing, particle systems, audio — a script tag and nothing else). No C++, no Unreal Engine, no engine purchase, no compile step for the game itself — but the *deliverable* is native binaries, so packaging is a real step (below). Only use a different game engine (Three.js, Godot, C++/UE) if the user explicitly names it *and* their machine can build/run it. If the machine is weak (integrated GPU only, shared RAM — check the specs), bake the device into the brief: instancing, LOD, capped post-processing, resolution scaling, object budgets.

**Native packaging (when the user wants `.exe` / `.apk` instead of a browser tab):** the game core stays Babylon.js + procedural code (zero external assets) — packaging wraps it, it does not rewrite it.

- **Windows `.exe` — Tauri 2 first.** The same web game ships as one small native `.exe` (~5–15 MB) plus WebView2 loader `.dll` and a `config.ini` the game reads/writes (settings are code-authored, so the file is generated, not shipped). Requires the Rust toolchain + MSVC build tools on the machine; it is a one-time install, far lighter than any engine editor.
- **Android `.apk` — Tauri 2 mobile.** The same codebase, one `tauri android init` + build, produces a signed `.apk` that runs the game in the platform WebView. No Java engine, no Unity, no GDK.
- **Electron fallback.** If the machine cannot install Rust/MSVC, Electron packages the identical code into a `.exe` installer via npm (heavier binary, but zero new toolchains — Node is usually already present).
- **The packaging honesty rule:** a wrapper changes the window, not the workload — if the game hangs in the browser on an integrated GPU, the identical GPU work will also strain the wrapped `.exe`/`.apk`. What makes it not hang is the *perf budget baked into the brief from round one*: instancing, LOD, capped post-processing, resolution scaling, object budgets. Build the game to the perf floor (e.g. 30+ FPS at 720p on the iGPU) and the packaged binary inherits it.
- **Perf critic owns this:** every round, the perf critic measures frame rate in the shipping frame and rejects any round that drops below the floor — a pretty round that fails perf fails the round. The packaged binary is then just the same build inside a native shell.
- **The compiler ninja owns the pipeline:** in native-deliverable runs, the packaging work is not an end-of-run chore — it is a role that packages **every wave** (see COMPILER NINJA in Roles). The user's `.exe`, `.dll`, `.ini`, and `.apk` exist and are runnable from wave one, so "single binary at the end" is never a cliff — it is how every wave ships.

### Photoreal-tier bars (GTA VI-class pull)

When the user wants graphics beyond the previous title ("better than GTA V, at GTA VI level"), use a **two-bar structure** — this is how you make an unreachable bar concrete instead of a slogan:

- **The pull bar (unreachable, sets direction):** real GTA VI screenshots/footage — 4K stills from the trailer or press material, frozen into `bar/` before round one and hashed. The loop never beats it; that is the point. It exists so the critics never let the work settle at "pretty good for a browser game."
- **The floor bar (reachable, must win):** the highest-quality GTA V screenshots available — **best-in-class modded GTA V** (ENB/ReShade/NVE photoreal mods, 4K), comparable scenes (same city-view, same time of day). The conformance/regression gates run this one: every round, a critic blind A/B's the game against the modded shot, and a round only passes if ours wins or ties. Floor beat + pull chased = "GTA V ke saare mods fail, GTA VI ki taraf."
- State the structure to the user in one line: "The bar is GTA VI; the floor is the best modded GTA V. We do not stop until we beat the floor, and we keep pulling toward the ceiling until you stop us."

**What actually moves a browser render toward photoreal (give these to the builders):** PBR material chain (metalness/roughness maps generated procedurally from noise — per-pixel shader detail instead of texture files); ACES tone mapping + exposure; post-processing stack (bloom, SSAO, motion blur, vignette, chromatic aberration at edges, film grain); shadow quality (PCF soft shadows or cascaded shadow maps; contact shadows on characters/vehicles); image-based lighting from a procedurally generated environment map (gradient sky + sun disc + ground bounce); procedural water shader (normal perturbation, specular sun glint, shoreline foam); atmosphere (exponential fog, sun-scatter tint at horizon, light shafts); day/night cycle with temperature-shifted lighting; reflection probes for car paint; instanced vegetation density + wind shader. The critics' numbers: blind A/B vs the frozen GTA VI stills, plus measurable deltas — color histogram match, shadow softness, reflection sharpness, average luminance, frame rate in the shipping viewport.

**Honesty framing (say it once, then run):** the loop is the mechanism that squeezes the last drop of quality out of a machine — that is its entire job, and "nothing is impossible" is the right attitude to bring to it. Two constraints are physics, not pessimism: a browser renderer cannot out-render a 2025+ AAA engine, and an integrated-GPU machine has a lower photoreal ceiling than a 3090. So: the pull bar (GTA VI) stays unreachable by design, and the floor bar (best modded GTA V) is the real fight — every round it must be beaten or tied, and each win is a genuine "fails all the mods" screenshot on *this* device. The deliverable is the most photoreal GTA-class open world this loop can produce on this hardware with zero external assets, pulled toward GTA VI every round. The run is long — months is fine — and it keeps going until marginal-gain collapse (two consecutive rounds with no critic-visible improvement) or the user stops it; stopping is the normal ending, never a failure.

### Game critics — how a critic actually inspects a game

- **Visual critics** (one lens each: composition, materials, lighting, geometry, HUD): screenshot the running game at a fixed camera angle and frame, put the screenshot next to the frozen reference blind, and pick. Same aspect, size, and camera every round or no two numbers are comparable. If critics have no way to obtain a number, **build the measuring tool first** (a screenshot harness, a pixel-diff script).
- **Playability critic**: actually plays the game — inputs, physics feel, game loop, collision, menus, win/lose states. A beautiful non-playing game fails.
- **Perf critic**: frame rate in the shipping viewport, load time, bundle size. Games that chug lose to games that are smooth.
- **Audio critic**: sounds exist, don't clip, have variety, react to events. (PARAD111GM warns: audio is a domain where judges are unreliable — if you cannot trust a critic on sound, make it a human gate.)
- **One critic with no lens at all. Always.** In the Kart Royale run, a single unlensed critic found eight things four specialised critics had all missed — including that the road was simply the wrong shape. Lenses have a blind spot exactly where they meet.

### Reference-image bars (the concept-art pattern)

When the user supplies a reference image (a drawing, concept art, a photo of the game they want):

- The image **is** the bar. Freeze it into `bar/` before round one, hash it, and never re-fetch.
- The critic screenshots the running game at a comparable angle/framing and blind A/B's the two images.
- **Blockout before assets, in 3D too**: rebuild the reference's composition as flat-colored boxes at true size and position first. No model, texture, or light rescues a layout that is wrong, and a beautiful asset in the wrong place is worse than a box in the right place — it invites you to stop looking. In the vibegameengine worked example, nine grey rectangles caught a coordinate-origin error that every later measurement would have carried.

## Fan out critics, not just builders (vibegameengine)

- Run critics in parallel, read-only, **one lens each** (composition, materials, lighting, code correctness, perf, UX…), each explicitly told *not* to comment on the others' lenses — overlap produces four vague reviews instead of four sharp ones.
- **Demand numbers, not adjectives.** "Too dark" is unusable; "our midtones are rgb(93,97,78), the reference is rgb(136,95,77)" is a patch you can apply. If the critics have no way to obtain a number, build the measuring tool first.
- **Always run one critic with no lens at all.** Lenses have a blind spot exactly where they meet; the unlensed critic finds what the lensed ones all miss.
- **Make critique a separate, written act before touching code.** An agent that builds and judges in one motion is only checking that the code did what was typed.
- **Give every builder a file set it exclusively owns**, plus an explicit list of files another agent is editing right now. Two agents in one file lose work silently.
- **Let builders overrule critics, and make them report what they rejected.** A measured number is still a guess about intent; this is how the loop catches its own overcorrections — in the worked example, round 2 *reversed* a round-1 change instead of stacking a second fix on top of it.
- **Verify in the shipping frame, and cap the rounds.** Same aspect, size, and measuring conditions every round, or no two numbers are comparable. Write a round cap into the bar before starting: "until the critics go quiet" is the stop condition, the cap is the stop guarantee, and a defect whose cause is unfixable must be closed in writing or a naive critic will re-file it forever.

## Self-lint the emitted prompt (c2c8)

Before emitting a run prompt, check it against this floor — the community's linter catches a prompt that *forgot* a gate, not one that mentions it insincerely, but 13/13 is the entry bar:

1. Bar is named, not a category. ("Call of Duty screenshots" yes; "AAA quality" no.)
2. Bar is fetchable — the agent can obtain it now.
3. Bar is comparable — a blind A/B is imaginable.
4. Reference acquisition is instructed ("get the real thing first").
5. Goal given, not implementation — no architecture or stack spelled out (unless demanded).
6. Decomposition delegated to the agent.
7. Builder and critic are separate roles.
8. Critic is blind — no builder history, no builder summary.
9. Critic inspects the real artifact, not a description.
10. Harsh binary job — a forced pick, not a score.
11. Loop continues until the critic picks ours — no fixed round count.
12. Stop conditions exist (win, stall, budget, human).
13. A live progress page is requested.

## Meta-prompt: let a model write the run prompt (Shumer)

When you are not sure what the bar should be, or want the strongest possible run prompt, use the generator pattern from the original article instead of hand-writing it:

```
I want to run a Gauntlet Loop for this goal: [GOAL]

Possible references or quality bars: [OPTIONAL]

Choose the strongest concrete bar that an agent can actually inspect and compare
its work against. If I have not supplied one, propose a useful comp or measurement
that plays the same role for this task that real Call of Duty screenshots played
for Matt Shumer's Claude of Duty game. Explain the bar in one sentence.

Then write a short prompt for the agent in the style of Matt's prompt (minimal is
better — the agent should decide the specifics). Give the lead agent the goal and
the bar, but let it choose the approach. Tell it to divide the goal into the
smallest pieces that can be improved and judged independently. For each important
piece, it should fan out a builder and a separate critic with fresh context.

Each critic must inspect the real output, compare it directly with the bar — using
a blind A/B comparison when possible — identify the biggest remaining gap, and send
it back for another round. Keep looping until our output wins or I stop the run.

Have the lead agent maintain a simple live progress page that shows the work
evolving over time. Do not prescribe the architecture, exact decomposition, or a
fixed number of rounds. Keep the final prompt short, just like Matt's.
```

## Cost and budget

The loop is expensive by design — that is the point (it spends compute on quality). Plan it instead of discovering it:

- Rough math: 5 units × 4 rounds ≈ 40 agent invocations. Decide the budget on purpose.
- **Judging dominates, not building.** Round-close panels (the whole against the bar, and this round against last) cost more than the builders. Never economize on the critic — **a cheap critic is a captured critic**. Cheap builders + expensive critic cuts roughly an order of magnitude at little quality cost; never the reverse.
- Set every model/effort level **at spawn** — a resumed agent reverts to defaults.
- Put the budget in the harness or the stop line, **never in the run prompt** — a round counter there competes with the bar, and the counter wins.
- Do not price the stop line in dollars; name a ceiling in rounds/time/tokens.

## Prompt template (when emitting a run prompt)

Adapt the wording every time. Fill the brackets, keep it short (120–180 words), keep the last line. No bullets inside the prompt; it should read like someone telling an agent what perfect looks like and refusing to accept less.

```
Build [GOAL].

The bar is [BAR]. Get the real thing first and compare against it directly, not against a description of it.

Break this into the smallest pieces that can be improved and judged on their own. For each piece, fan out a builder and a separate critic with fresh context. The critic inspects the actual output, puts it next to the bar blind with the labels stripped, says which one is better, and names the single biggest remaining gap. Then it goes back to the builder.

The critic should be a harsh critic. Praise is not useful. If ours does not win, it keeps going.

Keep looping until the critic picks ours blind. Do not stop before that. Run the builders and critics as parallel subagents.

Keep a live progress page updating as the work evolves so I can watch it.
```

Rules for what you fill in: bake the bar in as a concrete fetchable thing (URL, product name, repo, title, image file); add a budget/cost ceiling **only if the user named one**; add tool names only if the goal needs them; everything else stays out — no architecture, no decomposition, no round count, no stack choice unless demanded.

## Monitoring without interrupting

Maintain a **live progress workbench** (`workbench.md` or a self-refreshing page): current round, per-unit PASS/FAIL, critic evidence, links to latest artifacts/screenshots. Read it asynchronously; intervene only when the loop is stuck on the wrong thing. For long runs expect hours — do not poll the agent; the page is the interface.

## What breaks a gauntlet loop

- **A vague bar.** The critic invents a comparison and approves everything. Most common failure by far.
- **The builder judging its own work.** Critic must be separate, fresh context, no knowledge of the builder's effort.
- **A stale critic.** A critic that graded a previous draft then grades the retry grades *improvement*, not the bar. Fresh critic per round.
- **A soft critic.** Say "harsh" and give a binary job. Scores out of 10 drift upward every round.
- **Critics without measurement tools.** Five critics dispatched with a brief they have no means of satisfying come back with adjectives. Build the measuring tool first.
- **Named exit after N rounds.** The exit is winning the comparison, or the user stopping the run.
- **Over-specifying.** Every extra instruction is one fewer decision the agent makes with its own judgment.
- **No budget cap.** An unreachable bar with no ceiling cannot end. The one way to lose money on a good prompt.
- **Bar drift.** Re-fetching or re-interpreting the bar mid-run invalidates every comparison. Freeze and hash it at round zero.
- **Critic capture.** A critic that becomes agreeable, or one that never sees the real artifact, certifies everything.
- **A weak or wrong brief.** The loop is an amplifier — it optimizes hard toward the wrong thing very convincingly, and the conformance critic keeps it honest about the brief you *wrote*, not the brief you *meant*. If direction matters more than polish, do one ordinary pass first, fix the direction, then start the loop. This is a finishing tool at least as much as a starting one.

## When NOT to use this

Skip for small, low-stakes, one-off work (quick answers, throwaway scripts, one-line fixes). The loop costs many times the tokens and wall-clock of a single pass. Use it when quality genuinely matters and you can name something real to be measured against. Also skip when:

- **No external exemplar exists** (novel research, "figure out what we should build"). The loop optimizes toward a destination; it cannot choose one. Shape the goal first.
- **Correctness is defined by a spec or test suite.** A green test beats any critic. Run TDD and CI; keep the gauntlet for taste, feel, polish, craft.
- **Actions are irreversible or side-effectful** (sent messages, migrations, money, live calls). A frozen probe re-runs every round, so a probe that sends, sends every round.

## Kilo portability notes

- Kilo has no `/loop` or `ultracode`: run the builders and critics as parallel `task` subagents with clean contexts, and keep looping within the session until the critic picks ours or the user stops the run. In game mode, the lead can drive screenshots via browser tooling so critics judge real pixels, not claims.
- The user is the brake. The loop will not finish on its own — stop when the bar is beaten, progress stalls two consecutive rounds, the budget is spent, or the user calls it.
- Long runs span hours; keep the workbench updated so the user can watch, and report the round log + evidence at the end.

## Credits

Technique: Matt Shumer (Claude of Duty, somethingbig.ai/gauntlet-loop). Merged skill draws on: robonuggets/gauntlet-loop (CC BY 4.0), trilwu/gauntlet-loop-skills, duolahypercho/gauntlet-loop (MIT), NicholasSpisak/gauntlet-loop (MIT), vibegameengine/gauntlet-loop (MIT), c2c8/gauntlet-loop (CC BY 4.0), PARAD111GM/gauntlet-loop-system. Case studies: mshumer/Claude-of-Duty (original prompt) and Kart Royale (racing.ryancampbell.com, zero-external-asset browser kart racer).

## Sources

- https://somethingbig.ai/gauntlet-loop (the original write-up + prompt generator)
- https://github.com/mshumer/Claude-of-Duty (original prompt + open-sourced game)
- https://racing.ryancampbell.com/ (Kart Royale — single ~1.75 MB bundle, zero asset requests)
- https://github.com/robonuggets/gauntlet-loop
- https://github.com/trilwu/gauntlet-loop-skills
- https://github.com/duolahypercho/gauntlet-loop
- https://github.com/NicholasSpisak/gauntlet-loop
- https://github.com/vibegameengine/gauntlet-loop
- https://github.com/c2c8/gauntlet-loop
- https://github.com/PARAD111GM/gauntlet-loop-system


Base directory for this skill: C:\Users\thaku\.config\kilo\skills\gauntlet-loop
Relative paths in this skill (e.g., scripts/, references/) are relative to this base directory.

Abe, ab "TRANSCRIPT.md" padh and saara ka saara kaam, saare rules ko follow karte hue complete kar! Aur benchmarks jo hai unko hi exact hardcoded jo numbers hai unko exact real numbers me badal de! Sab real kar! No stubs! Lag jao, maximum sub-agents! Ye point yaad rakhna, maxmimum sub-agents use karne hai!