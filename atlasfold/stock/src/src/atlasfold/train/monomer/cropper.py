import numpy as np

from atlasfold.common import protein
from atlasfold.utils.misc import spawn_rng


class ProteinCropper:
    def __init__(
        self,
        prob_contiguous: float = 0.2,
        prob_spatial: float = 0.6,
        prob_multi_contiguous: float = 0.2,
        max_num_segments: int = 4,
        min_segment_length: int = 32,
    ):
        self.prob_spatial = prob_spatial
        self.prob_contiguous = prob_contiguous
        self.prob_multi_contiguous = prob_multi_contiguous
        self.max_num_segments = max_num_segments
        self.min_segment_length = min_segment_length

    def crop(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop the input structure

        Parameters
        ----------
        prot : protein.Protein
            The input protein structure to be cropped.
        rng : np.random.Generator
            The random number generator for reproducibility.

        Returns
        -------
        sequence_indices : np.ndarray
            The indices of the residues to be included in the cropped structure.
            NOTE: this is different from the residue index (1-based).
        """
        rng = spawn_rng(rng)
        if len(prot) <= max_length:
            return np.arange(len(prot))

        # Randomly choose cropping strategy
        r = rng.random()
        if r < self.prob_spatial:
            return self._crop_spatial(prot, max_length, rng)
        elif r < self.prob_spatial + self.prob_contiguous:
            return self._crop_contiguous(prot, max_length, rng)
        else:
            return self._crop_multi_contiguous(prot, max_length, rng)

    def _crop_contiguous(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a contiguous segment of the input structure."""
        length = len(prot)
        start = rng.integers(0, length - max_length + 1)
        return np.arange(start, start + max_length)

    def _crop_spatial(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop a spatial segment of the input structure."""
        length = len(prot)
        # CA coordinates are at index 1 in coordinates [L, 14, 3]
        ca_coords = prot.coordinates[:, 1, :]
        ca_mask = np.isfinite(ca_coords).all(axis=-1)
        valid_indices = np.where(ca_mask)[0]

        if not np.any(ca_mask):
            # Fallback to contiguous if no CA is found
            return self._crop_contiguous(prot, max_length, rng)

        if len(valid_indices) <= max_length:
            # If there are fewer valid CA residues than max_length,
            # return all valid indices
            return valid_indices

        # Pick a random anchor residue that has a CA
        anchor_idx = rng.choice(list(valid_indices))
        anchor_coord = ca_coords[anchor_idx]

        # Compute distances to all CA coordinates
        distances = np.full(length, 1e6, dtype=np.float32)
        valid_distances = np.linalg.norm(ca_coords[ca_mask] - anchor_coord, axis=-1)
        distances[ca_mask] = valid_distances

        # Break ties randomly by shuffling indices first
        indices = np.arange(length)
        rng.shuffle(indices)
        shuffled_distances = distances[indices]

        # Pick the max_length nearest residues
        nearest_shuffled_indices = np.argsort(shuffled_distances)[:max_length]
        nearest_indices = indices[nearest_shuffled_indices]

        return np.sort(nearest_indices)

    def _crop_multi_contiguous(
        self,
        prot: protein.Protein,
        max_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r"""Crop multiple contiguous segments of the input structure.

        Motivation: The training sample from the spatial cropping and contiguous
        cropping has a restricted variance in the input structure. Therefore,
        we introduce a multi-contiguous cropping strategy that allows for model
        to predict the large structure.

        Strategy:
            1. Randomly choose the number of segments (between 2 and max_num_segments).
            2. Uniformly segment the sequence into num_anchors segments.
            3. Randomly distribute max_length into crop_sizes.
            4. Crop contiguous residues from each uniform segment.
        """
        if max_length < self.min_segment_length * 2:
            # We need at least 2 segments to do multi-contiguous cropping
            return self._crop_contiguous(prot, max_length, rng)

        length = len(prot)
        # 1. Randomly choose the number of segments
        max_num_segments = min(self.max_num_segments, length // self.min_segment_length)
        num_segments = rng.integers(2, max_num_segments + 1)

        # 2. Segment the sequence
        segment_length = length // num_segments
        seg_sizes: list[int] = [segment_length] * num_segments
        # Add the remaining residues to the last segment
        seg_sizes[-1] = length - segment_length * (num_segments - 1)

        # 3. Randomized crop sizes
        budget = max_length - num_segments * self.min_segment_length
        assert budget >= 0
        crop_sizes = [self.min_segment_length] * num_segments
        valid_segments = [
            i for i in range(num_segments) if seg_sizes[i] > self.min_segment_length
        ]
        while budget > 0:
            # Randomly pick a segment to add one more residue to
            idx = rng.choice(valid_segments)
            max_len = seg_sizes[idx]
            curr_len = crop_sizes[idx]
            if curr_len < max_len:
                # Add a random number of residues to this segment.
                max_to_add = min(budget, max_len - curr_len)
                num_to_add = rng.integers(1, max_to_add + 1)
                crop_sizes[idx] += int(num_to_add)
                budget -= int(num_to_add)
            if crop_sizes[idx] >= max_len:
                # This segment cannot be added anymore, remove it from valid segments
                valid_segments.remove(idx)
                if not valid_segments:
                    break

        current_idx = 0
        cropped_indices = []
        for i in range(num_segments):
            # Crop a contiguous segment from this segment
            seg_len = seg_sizes[i]
            crop_len = crop_sizes[i]
            start_in_seg = rng.integers(0, seg_len - crop_len + 1)
            start = current_idx + start_in_seg
            cropped_indices.append(np.arange(start, start + crop_len))

            current_idx += seg_len

        return np.sort(np.concatenate(cropped_indices))
