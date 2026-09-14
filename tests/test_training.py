import pytest
import torch
from torch import nn

from earnings_asr.training import acoustic_lora_targets, apply_acoustic_lora, mask_target_labels


@pytest.mark.parametrize("left", [True, False])
def test_only_target_tokens_supervised_with_variable_lengths(left):
    # EOS ID also appears in padding: attention_mask must determine which to keep.
    if left:
        full = {"input_ids": torch.tensor([[1, 2, 3, 9], [9, 1, 7, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])}
        prefix = {"input_ids": torch.tensor([[1, 2], [9, 1]]),
                  "attention_mask": torch.tensor([[1, 1], [0, 1]])}
        expected = [[-100, -100, 3, 9], [-100, -100, 7, 9]]
    else:
        full = {"input_ids": torch.tensor([[1, 2, 3, 9], [1, 7, 9, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])}
        prefix = {"input_ids": torch.tensor([[1, 2], [1, 9]]),
                  "attention_mask": torch.tensor([[1, 1], [1, 0]])}
        expected = [[-100, -100, 3, 9], [-100, 7, 9, -100]]
    assert mask_target_labels(full, prefix).tolist() == expected


def test_unexpected_prompt_tokens_fail_closed():
    with pytest.raises(ValueError, match="exact prefix"):
        mask_target_labels({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)},
                           {"input_ids": torch.tensor([[7]]), "attention_mask": torch.ones(1, 1)})


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 4)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


class TinyAudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.proj1 = nn.Linear(4, 4)
        self.proj2 = nn.Linear(4, 4)


class TinyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = TinyAudioTower()
        self.model = TinyAttention()
        self.lm_head = nn.Linear(4, 4)


class TinyASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = TinyThinker()

    def forward(self, values):
        return self.thinker.lm_head(values)


def test_acoustic_lora_matches_only_audio_attention_and_projector():
    model = TinyASR()
    targets = acoustic_lora_targets(model)
    assert len(targets) == 8
    assert all(name.startswith("thinker.audio_tower.layers.") for name in targets)

    adapter, report = apply_acoustic_lora(
        model, {"lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0})
    trainable = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]

    assert report["decoder_frozen"]
    assert any("proj1.modules_to_save" in name for name in trainable)
    assert any("proj2.modules_to_save" in name for name in trainable)
    assert not any("thinker.model" in name or "thinker.lm_head" in name for name in trainable)


@pytest.mark.parametrize("projector_mode", ["frozen", "lora"])
def test_projector_training_modes(projector_mode):
    adapter, report = apply_acoustic_lora(
        TinyASR(),
        {"lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0,
         "projector_mode": projector_mode},
    )
    trainable = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]

    assert report["projector_mode"] == projector_mode
    assert report["decoder_frozen"]
    if projector_mode == "frozen":
        assert not any("audio_tower.proj1" in name or "audio_tower.proj2" in name
                       for name in trainable)
    else:
        assert any("audio_tower.proj1.lora_" in name for name in trainable)
        assert any("audio_tower.proj2.lora_" in name for name in trainable)
        assert not any("modules_to_save" in name for name in trainable)


def test_unknown_projector_mode_fails_closed():
    with pytest.raises(ValueError, match="projector_mode"):
        apply_acoustic_lora(
            TinyASR(),
            {"lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0,
             "projector_mode": "unknown"},
        )
