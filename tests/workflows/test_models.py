from lovelaice.workflows import AgentNode, PromptNode, SequenceNode, WorkflowSpec


def test_parse_sequence_of_agents_from_dict():
    spec = WorkflowSpec.model_validate(
        {
            "name": "daily",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "agent", "prompt": "gather", "name": "raw"},
                    {"kind": "agent", "prompt": "shape {raw}", "output_schema": {"type": "object"}},
                ],
            },
        }
    )
    assert spec.name == "daily"
    assert isinstance(spec.root, SequenceNode)
    assert isinstance(spec.root.children[0], AgentNode)
    assert spec.root.children[0].name == "raw"
    assert spec.root.children[1].output_schema == {"type": "object"}


def test_bare_agent_root():
    spec = WorkflowSpec.model_validate(
        {"name": "one", "root": {"kind": "agent", "prompt": "hi"}}
    )
    assert isinstance(spec.root, AgentNode)
    assert spec.root.name is None


def test_prompt_node_parses_via_discriminator():
    spec = WorkflowSpec.model_validate(
        {
            "name": "wf",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "agent", "prompt": "draft a haiku"},
                    {"kind": "prompt", "prompt": "critique the haiku above", "name": "crit"},
                ],
            },
        }
    )
    child = spec.root.children[1]
    assert isinstance(child, PromptNode)
    assert child.prompt == "critique the haiku above"
    assert child.name == "crit"
