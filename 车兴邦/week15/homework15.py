import os, sys, time, re, json, uuid, argparse, logging, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ============================================================
# 0. 配置
# ============================================================

LLM_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
TAVILY_URL = "https://api.tavily.com/search"

MOCK = bool(os.getenv("MOCK", ""))          # 也可用环境变量开 mock
MAX_SUBAGENTS = 6                            # 单次派发子 agent 数上限（保护）
SUB_RESULT_CHARS = 500                       # 子结果回灌主 agent 的截断长度

# ============================================================
# 1. LLM 客户端（真实 DeepSeek / OpenAI 兼容接口；可切 Mock 离线演示）
# ============================================================

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI          # 延迟 import：不开 key 也能用 --mock
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("缺少 DEEPSEEK_API_KEY；无 key 请加 --mock 离线演示")
        _client = OpenAI(api_key=key, base_url=LLM_URL)
    return _client


def _mock_chat(system: str, user: str, **kw) -> str:
    """无 key 时的离线 LLM，用于演示整套派发/并行机制。

    只服务于"能跑通流程"，不做真推理；输出是脚本化的 ReAct 文本。
    用 sleep 模拟子 agent 的工作时长，让并行的 wall 与串行 sum 可测出差距。
    """
    is_sub = ("web_search" in system) and ("dispatch_subagents" not in system)
    question = user.split("\n")[0].replace("Question: ", "").strip()

    if is_sub:  # 子 agent：直接给结论（模拟它调研完返回），带一点随机耗时
        time.sleep(0.5 + (len(question) % 5) * 0.15)
        return ("Thought: 我已查完该子课题，直接汇总。\n"
                "Final Answer: " + mock_search_content(question))
    # 主 agent
    if "Observation:" not in user:          # 还没派发/搜索 -> 给出本轮动作
        subs = pick_subtopics(question)
        if len(subs) == 1:                  # 单一侧面 -> 走单次 web_search
            return (f"Thought: 单一事实问题，直接搜索一次即可。\n"
                    f"Action: web_search\nAction Input: {subs[0]}")
        return (f"Thought: 该问题含 {len(subs)} 个侧面，应派发子调研员并行收集。\n"
                f"Action: dispatch_subagents\n"
                f"Action Input: {' | '.join(subs)}")
    # 已拿到 Observation（子调研结果或单次搜索结果）-> 综合
    names = re.findall(r"【子课题: (.+?)】", user)
    if names:
        final = ("[mock] 综合报告：并行调研覆盖 " + "、".join(names)
                 + "，各侧面结论见上文（离线演示用）。")
    else:
        m = re.search(r"摘要: ([^\n]+)", user) or re.search(
            r"Observation: ([\s\S]{0,120})", user)
        final = "[mock] 单一事实结果：" + (m.group(1).strip() if m else "已按搜索结果回答")
    return f"Thought: 已收集足够信息。\nFinal Answer: {final}"


def chat(system: str, user: str, *, temperature: float = 0.0,
         max_tokens: int = 768, stop=None, retries: int = 3) -> str:
    """单轮 LLM 对话。MOCK 模式下走脚本化返回。stop 用于 ReAct 在 Observation 前截断。"""
    if MOCK:
        return _mock_chat(system, user, temperature=temperature,
                          max_tokens=max_tokens, stop=stop)
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens,
                stop=stop or None)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning("LLM 重试(%s): %s", attempt + 1, str(e)[:80])


# ============================================================
# 2. 工具：联网搜索（Tavily，urllib 零 SDK 依赖）
# ============================================================

def web_search(query: str, max_results: int = 5, **_kw) -> str:
    """返回给 LLM 的搜索结果文本。key 缺失 / 网络失败 -> 返回错误串，不抛异常。
    **_kw：吸收 ReAct 引擎固定传入的 shared_state，保持所有工具签名一致。"""
    if MOCK:
        return mock_search_content(query)
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return "搜索失败: 未设置 TAVILY_API_KEY（真实搜索需 key，或加 --mock 演示）"
    try:
        payload = {"api_key": key, "query": query, "max_results": max_results,
                   "search_depth": "basic", "include_answer": True}
        req = urllib.request.Request(TAVILY_URL,
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = [f"摘要: {data.get('answer')}"] if data.get("answer") else []
        for i, r in enumerate(data.get("results", [])[:max_results], 1):
            parts.append(f"[{i}] {r.get('title', '')}\n    {(r.get('content') or '')[:300]}")
        return "\n".join(parts) if parts else "无结果"
    except Exception as e:                       # ReAct 兜底，不打断主流程
        logger.warning("Tavily 搜索失败 '%s': %s", query, e)
        return f"搜索失败: {type(e).__name__}: {str(e)[:100]}"


def mock_search_content(query: str) -> str:
    """mock 搜索：按关键词给固定的"假资料"，让离线演示有 Observation 可看。"""
    q = query.lower()
    if any(k in q for k in ("销量", "规模", "装机", "size", "sale", "market")):
        return ("摘要: 市场规模数据见下（mock 数据）。\n"
                "[1] 市场规模报告\n    2024 年该市场整体规模约 X 亿元，同比增长约 Y%（示意值）。")
    if any(k in q for k in ("竞争", "格局", "厂商", "品牌", "player", "brand", "compe")):
        return ("摘要: 主要厂商与格局见下（mock 数据）。\n"
                "[1] 竞争格局分析\n    头部厂商 A/B/C 合计份额约 Z%，第二梯队以差异化竞争为主（示意值）。")
    if any(k in q for k in ("政策", "趋势", "消费", "补贴", "policy", "trend")):
        return ("摘要: 政策与趋势见下（mock 数据）。\n"
                "[1] 政策趋势\n    补贴退坡后转向技术导向，消费向品质化、智能化迁移（示意值）。")
    return "摘要: （mock 通用结果）\n[1] 通用资料\n    该主题暂无细化 mock 数据，仅演示流程。"


_MULTI_WORDS = ("调研", "分析", "报告", "概况", "竞品", "对比", "趋势", "政策")


def pick_subtopics(question: str) -> list[str]:
    """mock 主 agent 的拆解策略：从「A：X、Y、Z」里切多个侧面；拆不出且是多维问题
    就给通用三侧面；否则按单一事实直接返回整句（走单次 web_search）。

    真实模式里这步由 LLM 自己完成（MAIN_SYSTEM 引导）；mock 只是用规则代替。
    """
    q = (question or "").strip().strip("。.!！?？")
    for sep in ("：", ":"):
        if sep in q:
            base, _, tail = q.partition(sep)
            tail = tail.replace("调研", "").replace("分析", "")
            sides = [s.strip() for s in re.split(r"[、,，;；/]", tail) if s.strip()]
            if len(sides) >= 2:                       # 拆出的侧面够多 -> 该并行派发
                return [f"{base}：{s}" for s in sides[:MAX_SUBAGENTS]]
            if sides:                                 # 只有单侧面 -> 交给单次搜索即可
                return [f"{base}：{sides[0]}"]
            break
    # 没有罗列式冒号：看是否是"多维"问题，是则通用三侧面，否则按单一事实处理
    if any(w in q for w in _MULTI_WORDS):
        return [f"{q}：现状与规模", f"{q}：主要参与者与竞争", f"{q}：趋势与政策"]
    return [q]


# ============================================================
# 3. 通用 ReAct 引擎（主 agent 与 subagent 共用）
# ============================================================

REACT_SYSTEM = """你是市场调研助手，能用以下工具联网搜索调研。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""


def _parse(text: str):
    """解析 Thought/Action/Action Input。兜底：无 Action 但有实质文本 -> Final Answer。"""
    thought = ""
    m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text or "", re.S)
    if m:
        thought = m.group(1).strip()[:400]
    mfa = re.search(r"Final Answer:\s*(.*)", text or "", re.S)
    if mfa:
        return thought, "Final Answer", mfa.group(1).strip()
    ma = re.search(r"Action:\s*(.*)", text or "")
    mi = re.search(r"Action Input:\s*(.*)", text or "")
    if ma:
        return thought, ma.group(1).strip(), (mi.group(1).strip() if mi else "")
    if (text or "").strip():                     # LLM 常直接写报告不带 Final Answer 前缀
        return thought or "综合结果", "Final Answer", text.strip()
    return thought, "", ""


class ReActLoop:
    """通用 ReAct 循环。主 agent / 每个 subagent 各自实例化一个，差别只在 tools。"""

    def __init__(self, name: str, tools: dict, system_prompt: str = REACT_SYSTEM,
                 max_steps: int = 6):
        """tools: {tool_name: (callable(action_input, shared_state=None)->str, 描述)}"""
        self.name = name
        self.tools = tools
        self.max_steps = max_steps
        self._sys_tpl = system_prompt
        self.trace: list[dict] = []

    def run(self, question: str, shared_state: dict = None) -> dict:
        self.trace = []
        t0 = time.time()
        tools_desc = "\n".join(f"- {n}: {d}" for n, (_, d) in self.tools.items())
        system = self._sys_tpl.format(tools_desc=tools_desc)
        history = f"Question: {question}\n\n"
        final_answer = ""

        for idx in range(self.max_steps):
            llm_out = chat(system, history, max_tokens=768, stop=["Observation:"])
            thought, action, action_input = _parse(llm_out)
            step = {"idx": idx, "agent": self.name, "thought": thought,
                    "action": action, "action_input": action_input, "observation": None}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                break

            step["final"] = False
            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation[:800]
            step["done"] = True
            self.trace.append(step)
            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:                                     # max_steps 超限强制收尾
            final_answer = "（已达最大步数）" + (self.trace[-1]["observation"] or ""
                                               if self.trace else "")
            self.trace.append({"idx": self.max_steps, "agent": self.name,
                               "thought": "步数超限", "action": "Final Answer",
                               "action_input": final_answer, "observation": None,
                               "final": True})

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2)}

    def _exec_tool(self, action, action_input, shared_state) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools)}"
        fn, _ = self.tools[action]
        try:
            return str(fn(action_input, shared_state=shared_state))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


# ============================================================
# 4. 主 agent：Orchestrator-Workers —— dispatch_subagents 并行派发
# ============================================================

MAIN_SYSTEM = """你是市场调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："新能源汽车市场调研：销量、竞争、政策" -> Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现：把 '课题1 | 课题2 | ...' 拆成 N 个 subagent，
    并行（ThreadPoolExecutor）或串行（for 循环）执行，收齐后返回汇总文本。

    返回文本会带上量化统计：wall_clock(并行墙钟) vs 各子 agent 时长之和(串行基线)，
    让主 agent（和人）都能看到「并行把 sum 压到 ≈ max」的证据。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:MAX_SUBAGENTS]
    if not subtopics:
        return "未解析出子课题，请用 | 分隔。"

    shared = shared_state if shared_state is not None else {}
    shared.setdefault("subagents", {})
    shared.setdefault("parallel_stats", [])
    shared.setdefault("dispatches", [])

    # 每个 subagent 都是 ReAct 循环，只有 web_search 一个工具
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(name=sid,
                        tools={"web_search": (web_search, "联网搜索，参数=查询词")},
                        max_steps=4)
        defs.append((sid, sub, topic))
    shared["dispatches"].append({"subtopics": subtopics,
                                 "subagent_ids": [sid for sid, _, _ in defs]})

    def _run_one(sid, sub, topic):                      # 每个子 agent 独立返回其结果
        return sid, topic, sub.run(topic, shared_state=shared)

    t0 = time.time()
    outcomes = {}                                       # sid -> (topic, result)
    if serial:                                          # 串行基线（A/B 用）
        for sid, sub, topic in defs:
            sid, topic, result = _run_one(sid, sub, topic)
            outcomes[sid] = (topic, result)
    else:                                               # 并行（核心价值所在）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid
                    for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid = futs[fut]
                sid, topic, result = fut.result()
                outcomes[sid] = (topic, result)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(result["duration"]
                           for _, result in outcomes.values()), 2)
    speedup = round(serial_sum / wall, 2) if wall > 0 else 0.0
    shared["parallel_stats"].append({"n_subagents": len(defs),
                                     "wall_clock": wall,
                                     "serial_sum": serial_sum,
                                     "speedup": speedup})

    # 收集子结果进 shared_state（供报告/A-B 用），并截短回灌主 agent 防 context 过大
    parts = []
    for sid, (topic, result) in outcomes.items():
        shared["subagents"][sid] = {"subtopic": topic,
                                    "final_answer": result["final_answer"],
                                    "duration": result["duration"]}
        parts.append(f"【子课题: {topic}】(用时{result['duration']}s)\n"
                     f"{result['final_answer'][:SUB_RESULT_CHARS]}")
    return (f"并行调研完成：{len(outcomes)} 个子调研员，wall-clock {wall}s"
            f"（串行需 {serial_sum}s，加速 {speedup}x）\n\n"
            + "\n\n".join(parts))


def run_research(question: str, serial: bool = False) -> dict:
    """执行一次调研。serial=True 时 subagent 串行（A/B 基线）。
    返回 {final_answer, main_trace, subagents, dispatch_actions}。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        shared_state = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=shared_state,
                                   serial=serial)

    main = ReActLoop(
        name="main",
        tools={
            "web_search": (web_search, "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
        },
        system_prompt=MAIN_SYSTEM,
        max_steps=8)
    result = main.run(question, shared_state=shared_state)
    return {"final_answer": result["final_answer"],
            "main_trace": result["trace"],
            "subagents": shared_state["subagents"],
            "dispatches": shared_state["dispatches"],
            "parallel_stats": shared_state["parallel_stats"]}


# ============================================================
# 5. 输出 / 演示 / 计时对比
# ============================================================

def demo_query() -> str:
    return ("2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势")


def _print_result(q: str, r: dict, tag: str = ""):
    print(f"\n{'='*70}")
    print(f"[{tag or '结果'}] 问题: {q[:60]}")
    acts = [s["action"] + (f"({s['action_input'][:30]})" if s["action_input"] else "")
            for s in r["main_trace"]]
    print(f"主 agent 动作: {' -> '.join(acts) or '(空)'}")
    if r.get("dispatches"):
        print(f"派发 {len(r['dispatches'])} 次")
    for sid, info in r["subagents"].items():
        print(f"  |-- [{sid}] {info['subtopic'][:36]} (用时{info.get('duration', '?')}s)\n"
              f"  |    final: {info['final_answer'][:80].replace(chr(10), ' ')}")
    stats = r.get("parallel_stats") or []
    if stats:
        s = stats[-1]
        print(f"  并行统计: {s['n_subagents']} 个子任务  wall={s['wall_clock']}s  "
              f"串行和={s['serial_sum']}s  加速 {s['speedup']}x")
    print(f"\n报告（前 240 字）:\n{r['final_answer'][:240]}")


def run_ab(q: str) -> dict:
    """同题跑 串行 vs 并行，量化 dispatch 加速比（演示 Amdahl 教学的硬数据）。"""
    from concurrent.futures import ThreadPoolExecutor
    from time import perf_counter

    def _measure(serial: bool):
        t0 = perf_counter()
        r = run_research(q, serial=serial)
        return r, round(perf_counter() - t0, 2)

    print("\n正在跑 串行 基线 ...")
    r_ser, ser_wall = _measure(serial=True)
    print("正在跑 并行 ...")
    r_par, par_wall = _measure(serial=False)
    n_sub = len(r_par["subagents"])
    print(f"\n{'='*70}\nA/B：并行 vs 串行（问题含 {n_sub} 个已派发子任务）")
    print(f"  并行墙钟 {par_wall}s | 串行墙钟 {ser_wall}s | 总墙钟加速 "
          f"{ser_wall/par_wall if par_wall else float('inf'):.2f}x")
    # 子任务自身耗时之和：从 trace 累加不可行(跨线程)，以串行墙钟近似子任务串行和
    return {"serial_wall": ser_wall, "parallel_wall": par_wall,
            "total_speedup": (ser_wall / par_wall) if par_wall else None,
            "parallel": r_par}


def run_jobs(questions: list[str]) -> list[dict]:
    """并行完成多项独立工作：每个问题一个主 agent，ThreadPool 并行跑。"""
    results = []
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futs = {pool.submit(run_research, q): q for q in questions}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                results.append((q, fut.result()))
            except Exception as e:                    # noqa: BLE001
                results.append((q, {"final_answer": f"该任务失败: {e}",
                                    "main_trace": [], "subagents": {}, "dispatches": []}))
    return results


def main():
    ap = argparse.ArgumentParser(description="作业15：可下发 subagent 的并行 agent（单文件）")
    ap.add_argument("--query", default=demo_query(), help="单问题（默认内置示例）")
    ap.add_argument("--mock", action="store_true", help="离线演示：无需任何 key")
    ap.add_argument("--ab", action="store_true", help="同题跑 串行 vs 并行 A/B 对比")
    ap.add_argument("--jobs", default=None,
                    help="并行多项独立工作：多个问题用 | 分隔，各自跑一个主 agent")
    args = ap.parse_args()

    global MOCK
    if args.mock:
        MOCK = True

    if not MOCK and not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("[提示] 未检测到 LLM key。加 --mock 可离线演示整套机制；"
              "或设置 DEEPSEEK_API_KEY 做真实调研。")
        sys.exit(1)

    if args.jobs:
        qs = [s.strip() for s in args.jobs.split("|") if s.strip()]
        t0 = time.perf_counter()
        results = run_jobs(qs)
        print(f"\n并行完成 {len(results)} 项独立工作，总墙钟 {time.perf_counter()-t0:.2f}s")
        for i, (q, r) in enumerate(results, 1):
            _print_result(q, r, f"job{i}")
        return

    if args.ab:
        run_ab(args.query)
        return

    _print_result(args.query, run_research(args.query), "调研结果")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
