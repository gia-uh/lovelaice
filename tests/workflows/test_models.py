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


def test_parallel_and_map_parse_via_discriminator():
    from lovelaice.workflows import MapNode, ParallelNode

    spec = WorkflowSpec.model_validate({
        "name": "fan",
        "root": {"kind": "sequence", "children": [
            {"kind": "parallel", "name": "sums", "children": [
                {"kind": "agent", "prompt": "sum {doc} for eng", "name": "eng"},
                {"kind": "agent", "prompt": "sum {doc} for exec", "name": "exec"},
            ]},
            {"kind": "map", "over": "runtimes", "as": "rt", "name": "research",
             "node": {"kind": "agent", "prompt": "research {rt}"}},
        ]},
    })
    par = spec.root.children[0]
    mp = spec.root.children[1]
    assert isinstance(par, ParallelNode) and len(par.children) == 2
    assert isinstance(mp, MapNode) and mp.over == "runtimes" and mp.as_ == "rt"


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
