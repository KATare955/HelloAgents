"""C1 第一层：v1.2 think/nothink rescoreB 的 verify 三维口径分析（临时脚本，不入库）"""
import json
from collections import Counter, defaultdict

BASE = "/Users/kay/Desktop/hello-agents-family/HelloAgents/eval/eval_data/results/baseline"

def load(mode):
    p = f"{BASE}/qwen3-14b-{mode}/v1.2/baseline_qwen3-14b-{mode}_v1.2_20261104_rescoreB.json"
    return json.load(open(p))["results"]

both = {m: load(m) for m in ["think", "nothink"]}

for mode, results in both.items():
    print(f"\n{'='*70}\n【{mode}】n={len(results)}")
    fs = Counter(r["verify"]["final_score"] for r in results)
    print("final_score 分布:", dict(sorted(fs.items())))
    ok = [r["id"] for r in results if r["verify"]["final_score"] >= 1.0]
    print(f"verify 满分 {len(ok)}/{len(results)} = {len(ok)/len(results)*100:.1f}%")
    ss0 = results[0]["verify"]["sub_scores"]
    print("sub_scores 字段:", ss0 if isinstance(ss0, dict) else type(ss0))
    bad = [r for r in results if r["verify"]["final_score"] < 1.0]
    print(f"\nverify 未满分 {len(bad)} 个：")
    for r in bad:
        v = r["verify"]
        print(f"  {r['id']:<18} score={v['final_score']}  sub={v['sub_scores']}  tags={v['violation_tags']}  caps={v['caps_triggered']}  tc={r['tool_correct']}")
    tags = Counter()
    for r in results:
        for t in r["verify"]["violation_tags"] or []:
            tags[t] += 1
    print("violation_tags 分布:", dict(tags))
    caps = Counter()
    for r in results:
        for c in r["verify"]["caps_triggered"] or []:
            caps[c] += 1
    print("caps_triggered 分布:", dict(caps))

print(f"\n{'='*70}\n【对比】")
think_score = {r["id"]: r["verify"]["final_score"] for r in both["think"]}
nothink_score = {r["id"]: r["verify"]["final_score"] for r in both["nothink"]}
think_bad = {i for i, s in think_score.items() if s < 1.0}
nothink_bad = {i for i, s in nothink_score.items() if s < 1.0}
print("verify 未满分交集:", sorted(think_bad & nothink_bad))
print("think 独有未满分:", sorted(think_bad - nothink_bad))
print("nothink 独有未满分:", sorted(nothink_bad - think_bad))
swapped = {i: (think_score[i], nothink_score[i]) for i in think_score
           if (i in think_bad) != (i in nothink_bad)}
print("单边翻车 case（think_score, nothink_score）:", dict(sorted(swapped.items())))

print(f"\n【nothink 场景 × verify 口径】")
by_sc = defaultdict(lambda: [0, 0])
for r in both["nothink"]:
    by_sc[r["scenario"]][1] += 1
    if r["verify"]["final_score"] >= 1.0:
        by_sc[r["scenario"]][0] += 1
for sc, (ok, tot) in sorted(by_sc.items()):
    print(f"  {sc:<12} {ok}/{tot} = {ok/tot*100:.1f}%")
print(f"【nothink 类型 × verify 口径】")
by_ty = defaultdict(lambda: [0, 0])
for r in both["nothink"]:
    by_ty[r["type"]][1] += 1
    if r["verify"]["final_score"] >= 1.0:
        by_ty[r["type"]][0] += 1
for ty, (ok, tot) in sorted(by_ty.items()):
    print(f"  {ty:<8} {ok}/{tot} = {ok/tot*100:.1f}%")
