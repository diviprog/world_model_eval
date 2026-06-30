# Failure Diagnostic — `rt-1-x` on `google_robot_pick_coke_can`

**243 episodes** · variant-aggregation sweep

## Headline

- **Success rate: 46.5%**  (113/243 episodes)

## Where it breaks (failure phase)

| Phase | Failed episodes | Share of failures |
|---|---:|---:|
| grasp | 82 | 63% |
| transport | 27 | 21% |
| reach | 21 | 16% |

## What drives failure (by condition)

- **background:** worst at `base` — 56% failure (106/189)
- **lighting:** worst at `darker` — 63% failure (17/27)
- **distractor:** worst at `False` — 54% failure (103/189)
- **table texture:** worst at `cabinet2` — 93% failure (25/27)
- **object pose:** worst at `vertical` — 75% failure (61/81)

## Dominant failure clusters & what to collect

1. **base background, vertical object pose → grasp** (76% failure, 48/63 episodes)
2. **with no distractor, vertical object pose → grasp** (76% failure, 48/63 episodes)
3. **base lighting, vertical object pose → grasp** (75% failure, 47/63 episodes)
4. **base table texture, vertical object pose → grasp** (73% failure, 46/63 episodes)
5. **base background, cabinet2 table texture → grasp** (93% failure, 25/27 episodes)

### Recommendation

Collect demonstrations in **base background, vertical object pose** scenes, targeting the **grasp** phase (closing the gripper on the object). This cluster fails 76% of the time across 48 failed episodes — the single largest recoverable gap.
