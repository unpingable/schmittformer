import torch

from src.circuit_hybrid import EventClassifier, observations_from_events


def test_event_classifier_forward_shapes() -> None:
    classifier = EventClassifier()
    events = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    observations = observations_from_events(events, sigma=0.0)
    logits = classifier(observations)
    assert logits.shape == (2, 3, 3)
