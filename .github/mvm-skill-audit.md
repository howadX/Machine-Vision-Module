# MVM Skill Contract Audits

Use this file only when a public MVM surface changes without requiring prose
changes in either canonical skill. Add one entry in the same pull request:

```text
YYYY-MM-DD | <commit or PR> | reviewed surfaces | why skills remain accurate
```

Changing a canonical `mvm/skills/*/SKILL.md` in the same pull request also
satisfies the gate. The validator checks co-change only; this log records the
human semantic review that automation cannot prove.

2026-07-26 | semantic-first recap fix | `recap_cmd.rs` source resolution | skills already define semantic sessions as the default history store and flat logs as retired legacy output
