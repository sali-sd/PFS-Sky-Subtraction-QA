# PFS Sky-Subtraction QA

Quality-assurance tool for PFS/Subaru sky subtraction. For a set of `pfsMerged`
visits it measures, per SKY fiber and per arm (blue `b`, red `r`, NIR `n`):

- **Sky-line residuals** — flux-to-sky ratio at the strongest sky lines.
- **Continuum residuals** — fitted flux and sky continuum in clean wavelength windows.

The results are written to CSV and summarized in multi-page PDF reports.

## Outputs

For a visit range and collection, the script produces:

| File | Contents |
| --- | --- |
| `PFS_SkySubQA_SkyLineResiduals_*.csv` | Per-fiber sky-line residual % per arm |
| `PFS_SkySubQA_ContinuumResiduals_*.csv` | Per-fiber continuum flux/sky per arm |
| `PFS_SkySubQA_Summary_*.csv` | Per-visit medians, offset and spread scores |
| `PFS_SkySubQA_SkyLinePlots_*.pdf` | Box / line / histogram sky-line plots |
| `PFS_SkySubQA_ContinuumPlots_*.pdf` | Continuum box and line plots |
| `PFS_SkySubQA_MedianSpectra_*.pdf` | Median spectra for best / worst-spread / worst-offset visits |
| `SkySubQA_Log_*.txt` | Full run log |

## Requirements

- The LSST / PFS 2D DRP stack (`lsst.daf.butler`, `pfs.datamodel`, `pfs.drp.stella`)
- `numpy`, `pandas`, `matplotlib`

## Usage

Edit the `repo`, `collections`, and `cpu_cores` values in the `__main__` block of
[`skysubQA.py`](skysubQA.py), then run:

```bash
python skysubQA.py
```

By default every `pfsMerged` visit in the collection is processed; edit the `visits`
line to restrict the set.
