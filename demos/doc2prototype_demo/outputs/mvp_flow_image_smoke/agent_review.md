# Downstream Agent Review

## Workflow

- Workflow: `structured_output_to_downstream_agent`
- Backend: `deterministic agent workflow with template generator`
- Device: `CPU`
- Goal: Generate a Mermaid diagram from extracted flowchart nodes and edges.

## Input Summary

```json
{
  "document_type": "flowchart",
  "node_count": 6,
  "edge_count": 5
}
```

## Agent Steps

- `PlannerAgent` / `derive_downstream_plan` / `completed`: Prepared `mermaid_diagram` generation plan for `flowchart`.
- `GeneratorAgent` / `generate_artifact` / `completed`: Generated `mermaid_diagram` using deterministic agent workflow with template generator.
- `ReviewAgent` / `review_artifact_coverage` / `pass`: Reviewed generated artifact against structured extraction results.

## Review Result

- Status: `pass`
- Passed checks:
  - All 6 extracted nodes are represented in the generated diagram.
  - The structured input contains 5 directed edges for downstream diagram generation.
