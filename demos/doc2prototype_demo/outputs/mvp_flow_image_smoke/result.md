# Doc2Prototype MVP Run

## Command

`main.py examples/flowchart_sample.png --task flowchart --device CPU --output-dir outputs/mvp_flow_image_smoke`

## Input

- Task: `flowchart`
- Source: `examples/flowchart_sample.png`
- Parser: `PaddleOCR-VL OpenVINO`
- OpenVINO device: `CPU`
- OpenVINO version: `2026.1.0-21367-63e31528c62-releases/2026/1`
- OpenVINO model: `ov_paddleocr_vl_model`

## Timings

| Stage | Seconds |
| --- | ---: |
| model_load | 3.955 |
| openvino_inference | 13.358 |
| structure_extraction | 0.001 |
| generation | 0.000 |
| total | 17.575 |

## Structured Output

- Schema version: `doc2prototype.mvp.v1`
- Document type: `flowchart`
- Nodes: `6`
- Edges: `5`

```json
{
  "schema_version": "doc2prototype.mvp.v1",
  "document_type": "flowchart",
  "title": "Extracted Flowchart",
  "nodes": [
    {
      "id": "node_1",
      "label": "Start - User opens login page",
      "type": "start"
    },
    {
      "id": "node_2",
      "label": "Enter credentials",
      "type": "process"
    },
    {
      "id": "node_3",
      "label": "Validate input - is the form empty?",
      "type": "decision"
    },
    {
      "id": "node_4",
      "label": "Send authentication request",
      "type": "process"
    },
    {
      "id": "node_5",
      "label": "Success - show dashboard",
      "type": "process"
    },
    {
      "id": "node_6",
      "label": "End",
      "type": "end"
    }
  ],
  "edges": [
    {
      "from": "node_1",
      "to": "node_2"
    },
    {
      "from": "node_2",
      "to": "node_3"
    },
    {
      "from": "node_3",
      "to": "node_4"
    },
    {
      "from": "node_4",
      "to": "node_5"
    },
    {
      "from": "node_5",
      "to": "node_6"
    }
  ],
  "raw_analysis": "Login Flow\n\nStep 1: Start - User opens login page\nStep 2: Enter credentials\nStep 3: Validate input - is the form empty?\nStep 4: Send authentication request\nStep 5: Success - show dashboard\nStep 6: End\n\nStart -> Enter credentials\nEnter credentials -> Validate input\nValidate input -> Send authentication request\nSend authentication request -> Success\nSuccess -> End"
}
```

## Generated Artifact

- Type: `mermaid_diagram`
- Generation backend: `deterministic template`

```
flowchart TD
    node_1([Start - User opens login page])
    node_2[Enter credentials]
    node_3{Validate input - is the form empty?}
    node_4[Send authentication request]
    node_5[Success - show dashboard]
    node_6([End])
    node_1 --> node_2
    node_2 --> node_3
    node_3 --> node_4
    node_4 --> node_5
    node_5 --> node_6
```
