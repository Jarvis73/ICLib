import torch
import torch.nn.functional as F

LARGE_NUM = 1e9


def supervised_loss(logits, labels):
    """Compute mean supervised loss over local batch."""
    loss = F.cross_entropy(logits, labels)
    return loss


def contrastive_loss(hidden,
                     hidden_norm=True,
                     temperature=1.0):
    """Compute loss for model.

    Args:
        hidden: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.

    Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = F.normalize(hidden, dim=-1)
    hidden1, hidden2 = torch.split(hidden, hidden.shape[0] // 2, 0)
    batch_size = hidden1.shape[0]
    device = hidden.device

    labels = torch.arange(batch_size).to(device)
    masks = F.one_hot(torch.arange(batch_size), batch_size).to(device)

    logits_aa = torch.matmul(hidden1, hidden1.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM   # remove diagnal entries
    logits_bb = torch.matmul(hidden2, hidden2.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM   # remove diagnal entries
    logits_ab = torch.matmul(hidden1, hidden2.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1.T) / temperature

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
    loss = (loss_a + loss_b) / 2

    return loss, logits_ab, labels
