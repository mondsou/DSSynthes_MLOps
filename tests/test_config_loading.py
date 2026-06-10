from hr_mlops.config import load_config


def test_load_config_has_required_sections(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("source:\n  gold_hr_table: a\noutputs:\n  attrition_table: b\n", encoding="utf-8")

    cfg = load_config(config_file)

    assert "source" in cfg
    assert "outputs" in cfg
