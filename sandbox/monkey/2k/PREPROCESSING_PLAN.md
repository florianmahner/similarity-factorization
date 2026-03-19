# THINGS-2k Monkey Preprocessing Plan

## Data Structure

```
data/things-monkey/2k/
├── monkeyN_session1/                    # Already time-averaged (from monkey-dimensions)
│   ├── THINGS_normMUA_raw.mat           # (23910, 256) trials × IT channels
│   └── zi_list.csv                      # Stimulus labels per trial
│
├── monkeyN_session2/
│   ├── THINGS_normMUA_time_resolved.mat # (300, 27105, 1024) time × trials × all_ch
│   ├── THINGS_normMUA_raw.mat           # (27105, 256) time-averaged IT
│   └── zi_list.csv
│
├── monkeyF/
│   ├── THINGS_normMUA_time_resolved.mat # (300, 30000, 1024) time × trials × all_ch
│   ├── THINGS_normMUA_raw.mat           # (30000, 320) time-averaged IT
│   ├── Hub1-instance1_B001.ns6          # Raw Blackrock recording
│   └── zi_list.csv
│
└── things2_imgs.mat                     # Stimulus metadata (class names)
```

**Channel slices (from time-resolved to ROI):**
- MonkeyN: V1=0-512, V4=512-768, IT=768-1024
- MonkeyF: V1=0-512, IT=512-832, V4=832-1024

**Time windows for averaging:**
- V1: 25-125ms
- V4: 50-150ms
- IT: 75-175ms
- Baseline: -100 to 0ms

## Stimulus Mapping

| Recording | Stimuli | Trials | Trials/Stim |
|-----------|---------|--------|-------------|
| MonkeyN session1 | 1854 | 23,910 | ~13 |
| MonkeyN session2 | 2000 | 27,105 | ~14 |
| MonkeyF | 2000 | 30,000 | 15 |

**Key facts:**
- MonkeyN session1 has 1854 stimuli (subset of THINGS)
- MonkeyN session2 and MonkeyF have 2000 stimuli (same set)
- All 1854 from session1 are in session2/F
- 146 extra stimuli in session2/F (not in session1)
- Stimulus labels in `zi_list.csv` are object names (e.g., "aardvark", "zebra")
- These match THINGS image folder names in `/SSD/datasets/things/behav1854/`

---

## Preprocessing Pipeline

### Step 1: Time Averaging (raw → processed)

**Input:** Raw time-resolved data `THINGS2_*_MUA_trials.mat`
**Output:** Time-averaged data `THINGS_normMUA_raw.mat`

For MonkeyN session2 and MonkeyF:
1. Load raw data: shape (trials, time, channels)
2. Apply baseline subtraction: subtract mean of [-50, 0]ms
3. Average over response window: [75, 175]ms for IT
4. Normalize (z-score or match MonkeyN session1 normalization)
5. Save as `THINGS_normMUA_raw.mat` with keys `data_it`, `data_v1`, `data_v4`

**Verify:** Output shape matches (n_trials, n_channels)

### Step 2: Generate zi_list (if missing)

**Input:** Raw data or stimulus presentation logs
**Output:** `zi_list.csv` with one stimulus label per trial

For each recording:
1. Extract stimulus labels from raw data or metadata
2. Verify labels are THINGS object names
3. Save as single-column CSV (no header)

**Verify:**
- Number of rows = number of trials
- Labels match THINGS image folder names

### Step 3: Validate Alignment

Before combining data, verify:
1. Stimulus labels use same naming convention across recordings
2. Same stimulus → same image (check with things2_imgs.mat if needed)
3. Channel counts are as expected (IT: 256 for N, 320 for F)

### Step 4: Filter to Common Stimuli

For combining MonkeyN sessions:
1. Find intersection: 1854 stimuli in both sessions
2. Filter trials BEFORE any normalization
3. Save filtered trial indices for reproducibility

### Step 5: Z-Score Normalization

For each recording separately (after filtering):
1. Compute mean and std per channel across all trials
2. Z-score: `(x - mean) / std`
3. Verify: mean ≈ 0, std ≈ 1 per channel

### Step 6: Average Per Stimulus

For each recording:
1. Group trials by stimulus label
2. Average across trials → (n_stimuli, n_channels)
3. Verify: no NaN, mean ≈ 0

### Step 7: Compute Reliability

For MonkeyN (combining sessions):
1. Concatenate z-scored trials from both sessions
2. Split-half reliability with 1000 permutations
3. Fisher z-transform for averaging
4. Spearman-Brown correction
5. Threshold: keep channels with reliability > 0.3

### Step 8: Compute RSM

Using filtered, reliable channels:
1. Gaussian kernel with median heuristic
2. Verify: all positive, diagonal = 1

### Step 9: Run SRF

1. Select rank k (cross-validate or fixed)
2. Run until convergence
3. Visualize embedding with THINGS images

---

## Validation Checklist

### After Step 1 (Time Averaging)
- [ ] Output shape: (n_trials, n_channels)
- [ ] No NaN values
- [ ] Data range reasonable (not all zeros, no extreme outliers)

### After Step 2 (zi_list)
- [ ] Row count matches trial count
- [ ] All labels are valid THINGS object names
- [ ] Expected number of unique stimuli

### After Step 4 (Filtering)
- [ ] Same stimuli in all recordings being combined
- [ ] Trial counts reduced appropriately

### After Step 5 (Z-scoring)
- [ ] Mean per channel ≈ 0 (< 1e-6)
- [ ] Std per channel ≈ 1

### After Step 6 (Averaging)
- [ ] Shape: (n_stimuli, n_channels)
- [ ] Overall mean ≈ 0
- [ ] Correlation between recordings > 0.5

### After Step 7 (Reliability)
- [ ] Values in valid range [-1, 1]
- [ ] Reasonable number of channels above threshold

---

## Open Questions

1. **Time window:** Is [75, 175]ms correct for IT? What about V1/V4?

2. **Normalization:** What normalization was applied to MonkeyN session1?
   - Need to match for session2

3. **Channel alignment:** Are channels in the same order across sessions?
   - Same electrodes recorded?

4. **Why low correlation?** Current r=0.27 between sessions is concerning
   - Is this expected noise?
   - Different electrode placement?
   - Preprocessing mismatch?

---

## Files

| File | Purpose | Status |
|------|---------|--------|
| `preprocessing.py` | Utility functions | ✓ Done |
| `validate_data.py` | Validation pipeline | ✓ Done |
| `time_average.py` | Step 1: Time averaging | TODO |
| `compute_reliability.py` | Step 7: Reliability | Exists |
| `run_srf.py` | Steps 8-9: RSM + SRF | Exists |
