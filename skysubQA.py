import pandas as pd
import numpy as np
import sys
import gc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor, as_completed
from lsst.daf.butler import Butler
import pfs.datamodel as datamodel
from pfs.drp.stella.readLineList import ReadLineListTask
from pfs.drp.stella.fitContinuum import FitContinuumTask
from pfs.drp.stella import SpectrumSet
import warnings
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")


def _save_raster(pdf, dpi=300):
    """Rasterize the current figure at 300 dpi so the PDF scrolls quickly."""
    fig = plt.gcf()
    for ax in fig.axes:
        ax.set_rasterized(True)
    pdf.savefig(bbox_inches="tight", dpi=dpi)


# ==== SKY SUBTRACTION QA — UNIFIED ALL ARMS ====
class _Tee:
    """Mirror stdout and stderr to both the console and a log file."""
    def __init__(self, log_path):
        self._log    = open(log_path, 'w', buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stdout.flush()
        self._log.flush()
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    def __exit__(self, *_):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._log.close()


# ==== ARM DEFINITIONS ====
ARMS        = ['b', 'r', 'n']
ARM_COLORS  = {'b': 'royalblue', 'r': 'tomato', 'n': 'saddlebrown'}
ARM_LABELS  = {'b': 'Blue arm',  'r': 'Red arm', 'n': 'NIR arm'}
ARM_OFFSETS = {'b': -0.24,       'r': 0.0,       'n': 0.24}
BOX_WIDTH   = 0.22

# Sky line scanning windows per arm
SKYLINE_ARMS = {
    'b': {'wavl_start': 550,  'wavl_end': 640,  'box': 2.5},
    'r': {'wavl_start': 820,  'wavl_end': 950,  'box': 2.5},
    'n': {'wavl_start': 1000, 'wavl_end': 1250, 'box': 2.5},
}

# Continuum extraction ranges per arm (clean regions, avoids dichroic edges)
CONT_RANGES = {'b': (380, 650), 'r': (700, 890), 'n': (940, 1260)}

# Full physical arm display ranges for median spectra zoom panels
ARM_FULL_RANGES = {'b': (380, 650), 'r': (630, 970), 'n': (940, 1260)}

# Percentile bounds for arm zoom y-limits (NIR uses tighter window due to spikes)
ARM_YLIM_PCTS = {'b': (5, 95), 'r': (5, 95), 'n': (16, 84)}

INT_HALF_WIDTH  = 3 * 0.08   # ±3 pixels × 0.08 nm/pixel = 0.24 nm
MIN_SEP         = 0.5        # nm — minimum wavelength separation for sky line deduplication
VISITS_PER_PAGE = 50
BAD_ARM_THRESHOLD  = 0.8     # exclude arm if >80% of its pixels are flagged bad/missing
ZERO_FLUX_THRESHOLD = 0.5   # exclude arm if >50% of its flux values are zero


# ==== SKY LINE PEAK FINDER AND INTEGRATOR ====
def skyid(df, wavl, wavh):
    df_range = df[(df['Wavelength'] > wavl) & (df['Wavelength'] < wavh)]
    if df_range.empty or df_range['Sky'].isna().all():
        return None, np.nan, np.nan, np.nan
    df_peak = df_range.loc[df_range['Sky'].idxmax()]
    peak_center_wav = float(df_peak['Wavelength'])
    peak_sky = float(df_peak['Sky'])
    df_window = df[(df['Wavelength'] > peak_center_wav - INT_HALF_WIDTH) &
                   (df['Wavelength'] < peak_center_wav + INT_HALF_WIDTH)]
    integrated_flux = float(df_window['Flux'].sum())
    integrated_sky  = float(df_window['Sky'].sum())
    return peak_center_wav, peak_sky, integrated_flux, integrated_sky


# ==== TOP-10 SKY LINE SELECTOR WITH DEDUPLICATION ====
def select_top10(df, wavl_start, wavl_end, box):
    list_wavl = np.arange(wavl_start, wavl_end - box * 0.5, box)
    list_wavh = list_wavl + box

    wavskyline, skyline, fluxline, fluxtosky = [], [], [], []
    for wlow, whigh in zip(list_wavl, list_wavh):
        peak_wav, peak_sky, int_flux, int_sky = skyid(df, wlow, whigh)
        if peak_wav is None or np.isnan(peak_sky):
            continue
        wavskyline.append(peak_wav)
        skyline.append(peak_sky)
        fluxline.append(int_flux)
        if int_sky != 0 and not np.isnan(int_sky) and not np.isinf(int_flux / int_sky):
            fluxtosky.append(float((int_flux / int_sky) * 100))
        else:
            fluxtosky.append(np.nan)

    dfall = pd.DataFrame({'Wavsky': wavskyline, 'Sky': skyline,
                          'Flux': fluxline, 'Fluxtosky': fluxtosky})
    dfall_valid = dfall[np.isfinite(dfall['Fluxtosky'])]
    dfall_sorted = dfall_valid.sort_values('Sky', ascending=False).reset_index(drop=True)

    kept_wavs, kept_indices = [], []
    for i, row in dfall_sorted.iterrows():
        if all(abs(row['Wavsky'] - w) >= MIN_SEP for w in kept_wavs):
            kept_wavs.append(row['Wavsky'])
            kept_indices.append(i)
    return dfall_sorted.loc[kept_indices].nlargest(10, 'Sky')


# ==== UNIFIED PER-VISIT WORKER: SKY LINES + CONTINUUM FOR ALL ARMS ====
def skysub_all(visit, collections, repo):
    butler = Butler(repo, collections=collections)
    pfsConfig = butler.get('pfsConfig', dict(visit=visit))
    pfsMerged_visit = butler.get('pfsMerged', visit=visit)

    fiberid_all = [pfsConfig.fiberId[i] for i in range(len(pfsConfig))
                   if (pfsConfig.targetType[i] == datamodel.TargetType.SKY) and
                      (pfsConfig.fiberStatus[i] == datamodel.FiberStatus.GOOD)]

    config = ReadLineListTask.ConfigClass()
    config.lightSources = ["sky"]
    config.lampElements = []
    config.exclusionRadius = 1.0
    refLines = ReadLineListTask(config).run()
    refLines = refLines[(refLines.wavelength > 350) &
                        (refLines.wavelength < 1300) &
                        (refLines.intensity > 100) &
                        (refLines.status == 0)]

    task = FitContinuumTask(FitContinuumTask.ConfigClass())

    kept_fiberid      = []
    kept_spectrograph = []
    skyline_fib   = {arm: [] for arm in ARMS}
    cont_flux_fib = {arm: [] for arm in ARMS}
    cont_sky_fib  = {arm: [] for arm in ARMS}

    for fid in fiberid_all:
        try:
            pfsMerged = pfsMerged_visit.select(fiberId=fid)
            wav  = pfsMerged.wavelength[0]
            flux = pfsMerged.flux[0].copy()
            sky  = pfsMerged.sky[0].copy()
            spec = pfsMerged.spectrograph[0]

            pixels_bad = (pfsMerged.mask & pfsMerged.flags.get('BAD', 'CR', 'SAT', 'NO_DATA')) != 0
            flux[pixels_bad[0]] = 0.0
            sky[pixels_bad[0]]  = 0.0

            # Pixels with zero norm produce meaningless continuum values even if not
            # explicitly flagged bad — treat them as effectively bad for arm quality checks.
            norm_arr      = pfsMerged.norm[0]
            norm_zero     = ~np.isfinite(norm_arr) | (np.abs(norm_arr) < 1e-10)
            effectively_bad = pixels_bad[0] | norm_zero

            spectra_flux = SpectrumSet.fromPfsArm(pfsMerged)
            for s in spectra_flux:
                s.setFlux(flux)
            continuum_flux = task.run(spectra_flux, refLines)
            cFlux = (continuum_flux * pfsMerged.norm)[0]

            spectra_sky = SpectrumSet.fromPfsArm(pfsMerged)
            for s in spectra_sky:
                s.setFlux(sky)
            continuum_sky = task.run(spectra_sky, refLines)
            cSky = (continuum_sky * pfsMerged.norm)[0]

            sky_lines = sky - cSky
            plt.close("all")

            # NaN-mask bad pixels in the scanning arrays so the integration sums
            # in skyid exclude them; the zeroed flux/sky are kept only for the fits above.
            flux_scan = flux.copy().astype(np.float64)
            sky_scan  = sky_lines.copy().astype(np.float64)
            flux_scan[effectively_bad] = np.nan
            sky_scan[effectively_bad]  = np.nan
            df_skylines = pd.DataFrame({'Wavelength': wav, 'Flux': flux_scan, 'Sky': sky_scan})

            fib_skyline   = {}
            fib_contflux  = {}
            fib_contsky   = {}

            for arm in ARMS:
                lo, hi = CONT_RANGES[arm]
                mask_arm = (wav >= lo) & (wav <= hi)

                # Exclude arm if too many pixels are bad/zero-norm, or if flux is mostly zero
                if not mask_arm.any():
                    frac_bad  = 1.0
                    frac_zero = 1.0
                else:
                    n_pix     = mask_arm.sum()
                    frac_bad  = effectively_bad[mask_arm].sum() / n_pix
                    frac_zero = ((flux[mask_arm] == 0.0) & ~effectively_bad[mask_arm]).sum() / n_pix
                if frac_bad > BAD_ARM_THRESHOLD or frac_zero > ZERO_FLUX_THRESHOLD:
                    fib_skyline[arm]  = np.nan
                    fib_contflux[arm] = np.nan
                    fib_contsky[arm]  = np.nan
                    continue

                arm_cfg = SKYLINE_ARMS[arm]
                top10 = select_top10(df_skylines, arm_cfg['wavl_start'],
                                     arm_cfg['wavl_end'], arm_cfg['box'])
                fib_skyline[arm] = float(top10['Fluxtosky'].median()) if len(top10) > 0 else np.nan

                # Exclude bad/zero-norm pixels from the continuum median, matching the
                # masking used in the sky-line scan and the median-spectra merge.
                cont_mask = mask_arm & ~effectively_bad
                fib_contflux[arm] = float(np.nanmedian(cFlux[cont_mask]))
                fib_contsky[arm]  = float(np.nanmedian(cSky[cont_mask]))

            kept_fiberid.append(fid)
            kept_spectrograph.append(spec)
            for arm in ARMS:
                skyline_fib[arm].append(fib_skyline[arm])
                cont_flux_fib[arm].append(fib_contflux[arm])
                cont_sky_fib[arm].append(fib_contsky[arm])

        except Exception as e:
            plt.close("all")
            print(f"    Visit {visit} fiber {fid} skipped: {e}")
            continue

    return (visit, kept_fiberid, kept_spectrograph,
            skyline_fib, cont_flux_fib, cont_sky_fib,
            len(fiberid_all), len(kept_fiberid))


# ==== PARALLEL PROCESSING AND CSV OUTPUT ====
def process_visits(visits, collections, cpu_cores, repo):
    all_rows_skyline = []
    all_rows_cont    = []
    summary_results  = []
    visits_order     = visits
    collections_safe = collections.replace('/', '_')

    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = {executor.submit(skysub_all, visit, collections, repo): visit
                   for visit in visits_order}
        for future in as_completed(futures):
            visit = futures[future]
            try:
                (visit_done, fiberids, spectrographs,
                 skyline_fib, cont_flux_fib, cont_sky_fib,
                 fibers_attempted, fibers_kept) = future.result()

                fibers_dropped = fibers_attempted - fibers_kept
                print(f"  Visit {visit_done}: {fibers_kept}/{fibers_attempted} SKY fibers good"
                      f"  ({fibers_dropped} dropped)")

                for i, (fid, spec) in enumerate(zip(fiberids, spectrographs)):
                    row_sl = {"Visit": visit_done, "Fiberid": fid, "Spectrograph": spec}
                    row_ct = {"Visit": visit_done, "Fiberid": fid, "Spectrograph": spec}
                    for arm in ARMS:
                        row_sl[f"SkyLineResidual_pct_{arm}"] = skyline_fib[arm][i]
                        row_ct[f"FluxContinuum_{arm}"]        = cont_flux_fib[arm][i]
                        row_ct[f"SkyContinuum_{arm}"]         = cont_sky_fib[arm][i]
                    all_rows_skyline.append(row_sl)
                    all_rows_cont.append(row_ct)

                summary_results.append((
                    visit_done,
                    np.nanmedian(skyline_fib['b'])   if skyline_fib['b']   else np.nan,
                    np.nanmedian(skyline_fib['r'])   if skyline_fib['r']   else np.nan,
                    np.nanmedian(skyline_fib['n'])   if skyline_fib['n']   else np.nan,
                    np.nanmedian(cont_flux_fib['b']) if cont_flux_fib['b'] else np.nan,
                    np.nanmedian(cont_flux_fib['r']) if cont_flux_fib['r'] else np.nan,
                    np.nanmedian(cont_flux_fib['n']) if cont_flux_fib['n'] else np.nan,
                    np.nanmedian(cont_sky_fib['b'])  if cont_sky_fib['b']  else np.nan,
                    np.nanmedian(cont_sky_fib['r'])  if cont_sky_fib['r']  else np.nan,
                    np.nanmedian(cont_sky_fib['n'])  if cont_sky_fib['n']  else np.nan,
                ))

            except Exception as exc:
                print(f"  Visit {visit} FAILED: {exc}")

    summary_results.sort(key=lambda x: visits_order.index(x[0]))
    visit_str = f"{visits_order[0]}-{visits_order[-1]}"

    # --- Sky line data CSV ---
    skyline_file = f"PFS_SkySubQA_SkyLineResiduals_Visits={visit_str}_Collections={collections_safe}.csv"
    df_skyline = pd.DataFrame(all_rows_skyline).sort_values(by="Visit")
    sl_cols = ["Visit", "Fiberid", "Spectrograph"] + [f"SkyLineResidual_pct_{arm}" for arm in ARMS]
    df_skyline = df_skyline[sl_cols]
    df_skyline.to_csv(skyline_file, index=False, float_format="%.6f")

    # --- Continuum fiber data CSV ---
    continuum_file = f"PFS_SkySubQA_ContinuumResiduals_Visits={visit_str}_Collections={collections_safe}.csv"
    df_cont = pd.DataFrame(all_rows_cont).sort_values(by="Visit")
    ct_cols = ["Visit", "Fiberid", "Spectrograph"]
    for arm in ARMS:
        ct_cols += [f"FluxContinuum_{arm}", f"SkyContinuum_{arm}"]
    df_cont = df_cont[ct_cols]
    df_cont.to_csv(continuum_file, index=False, float_format="%.6f")

    # --- Summary CSV ---
    summary_file = f"PFS_SkySubQA_Summary_Visits={visit_str}_Collections={collections_safe}.csv"
    df_summary = pd.DataFrame({
        "Visit":                        [v[0] for v in summary_results],
        "SkyLineResidual_pct_median_b": [v[1] for v in summary_results],
        "SkyLineResidual_pct_median_r": [v[2] for v in summary_results],
        "SkyLineResidual_pct_median_n": [v[3] for v in summary_results],
        "FluxContinuum_median_b":       [v[4] for v in summary_results],
        "FluxContinuum_median_r":       [v[5] for v in summary_results],
        "FluxContinuum_median_n":       [v[6] for v in summary_results],
        "SkyContinuum_median_b":        [v[7] for v in summary_results],
        "SkyContinuum_median_r":        [v[8] for v in summary_results],
        "SkyContinuum_median_n":        [v[9] for v in summary_results],
    })

    # Visit-level scoring for median spectra visit selection (abs offset + spread across all arms)
    cont_abs_map    = {}
    cont_spread_map = {}
    for v in visits_order:
        df_v = df_cont[df_cont["Visit"] == v]
        all_vals = []
        for arm in ARMS:
            all_vals.extend(df_v[f"FluxContinuum_{arm}"].dropna().tolist())
        arr = np.array(all_vals)
        arr_finite = arr[np.isfinite(arr)]
        if len(arr_finite) > 0:
            cont_abs_map[v] = float(np.nanmedian(np.abs(arr_finite)))
        else:
            cont_abs_map[v] = np.nan
        if len(arr_finite) > 1:
            cont_spread_map[v] = float(np.percentile(arr_finite, 84) - np.percentile(arr_finite, 16))
        else:
            cont_spread_map[v] = np.nan

    df_summary["ContinuumFlux_median_abs"] = df_summary["Visit"].map(cont_abs_map)
    df_summary["ContinuumFlux_spread"]     = df_summary["Visit"].map(cont_spread_map)
    df_summary.to_csv(summary_file, index=False, float_format="%.6f")

    n_skyline = len(df_skyline["Visit"].unique())
    n_cont    = len(df_cont["Visit"].unique())
    n_summary = len(df_summary)

    print(f"\n{'='*60}")
    print(f"  Sky Fiber QA — Per-Visit Processing")
    print(f"{'='*60}")
    print(f"  Total visits requested : {len(visits_order)}")
    print(f"  Visits completed       : {len(summary_results)}")
    print(f"  Visits failed          : {len(visits_order) - len(summary_results)}")
    print(f"{'='*60}")
    print(f"  CSV Output files")
    print(f"{'='*60}")
    print(f"  Sky line data  ({n_skyline:3d} visits) : {skyline_file}")
    print(f"  Summary        ({n_summary:3d} visits) : {summary_file}")
    print(f"  Continuum data ({n_cont:3d} visits) : {continuum_file}")
    print(f"{'='*60}")

    return df_summary, skyline_file, summary_file, continuum_file, visits_order, collections_safe


# ==== SKY LINE PLOTS: BOX PLOTS, LINE PLOTS, AND HISTOGRAMS ====
def plot_skysub(skyline_file, collections, collections_safe, visits_order):
    df = pd.read_csv(skyline_file)
    visit_str = f"{visits_order[0]}-{visits_order[-1]}"
    pdf_name  = f"PFS_SkySubQA_SkyLinePlots_Visits={visit_str}_Collections={collections_safe}.pdf"

    valid_visits = []
    for visit in visits_order:
        df_v = df[df["Visit"] == visit]
        if len(df_v) == 0:
            continue
        if any(len(df_v[f"SkyLineResidual_pct_{arm}"].dropna()) > 0 for arm in ARMS):
            valid_visits.append(visit)

    arm_all_data = {arm: [] for arm in ARMS}
    for visit in valid_visits:
        df_v = df[df["Visit"] == visit]
        for arm in ARMS:
            data = df_v[f"SkyLineResidual_pct_{arm}"].dropna().values
            arm_all_data[arm].append(data if len(data) > 0 else np.array([np.nan]))

    n_valid = len(valid_visits)
    if n_valid == 0:
        print("  No sky line data to plot.")
        return

    # Y-limits for box plots: whisker-based, across all arms
    all_wlo, all_whi = [], []
    for arm in ARMS:
        for d in arm_all_data[arm]:
            d_c = d[np.isfinite(d)]
            if len(d_c) < 2:
                continue
            q1, q3 = np.percentile(d_c, [25, 75])
            iqr = q3 - q1
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            all_wlo.append(float(d_c[d_c >= lo_fence].min()) if (d_c >= lo_fence).any() else float(q1))
            all_whi.append(float(d_c[d_c <= hi_fence].max()) if (d_c <= hi_fence).any() else float(q3))
    if all_wlo and all_whi:
        pad_box = (np.percentile(all_whi, 98) - np.percentile(all_wlo, 2)) * 0.1
        ylim_lo_box = np.percentile(all_wlo, 2) - pad_box
        ylim_hi_box = np.percentile(all_whi, 98) + pad_box
    else:
        ylim_lo_box, ylim_hi_box = -10, 10

    # Per-arm per-visit stats for line plots
    medians = {arm: [float(np.nanmedian(d))      for d in arm_all_data[arm]] for arm in ARMS}
    p16s    = {arm: [float(np.nanpercentile(d, 16)) for d in arm_all_data[arm]] for arm in ARMS}
    p84s    = {arm: [float(np.nanpercentile(d, 84)) for d in arm_all_data[arm]] for arm in ARMS}

    all_p16 = [v for arm in ARMS for v in p16s[arm] if np.isfinite(v)]
    all_p84 = [v for arm in ARMS for v in p84s[arm] if np.isfinite(v)]
    if all_p16 and all_p84:
        pad_line     = (np.percentile(all_p84, 98) - np.percentile(all_p16, 2)) * 0.15
        ylim_lo_line = np.percentile(all_p16, 2)  - pad_line
        ylim_hi_line = np.percentile(all_p84, 98) + pad_line
    else:
        ylim_lo_line, ylim_hi_line = -10, 10

    chunks  = [list(range(i, min(i + VISITS_PER_PAGE, n_valid)))
               for i in range(0, n_valid, VISITS_PER_PAGE)]
    n_pages = len(chunks)

    legend_elements = [
        Patch(facecolor=ARM_COLORS[arm], edgecolor=ARM_COLORS[arm], alpha=0.7, label=ARM_LABELS[arm])
        for arm in ARMS
    ]

    with PdfPages(pdf_name) as pdf:

        # --- Box plots: 3 arms side by side per visit ---
        for page_idx, idx_chunk in enumerate(chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_pages})" if n_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(10, n * 1.0), 6))
            for arm in ARMS:
                positions  = [i + ARM_OFFSETS[arm] for i in range(n)]
                chunk_data = [arm_all_data[arm][i] for i in idx_chunk]
                box = ax.boxplot(chunk_data, positions=positions, widths=BOX_WIDTH,
                                 patch_artist=True,
                                 medianprops=dict(color='black', linewidth=1.5))
                for patch in box['boxes']:
                    patch.set_facecolor(ARM_COLORS[arm])
                    patch.set_alpha(0.6)
                    patch.set_edgecolor(ARM_COLORS[arm])
                for w in box['whiskers']: w.set_color(ARM_COLORS[arm])
                for c in box['caps']:     c.set_color(ARM_COLORS[arm])
                for f in box['fliers']:   f.set_markeredgecolor(ARM_COLORS[arm]); f.set_markersize(2)

            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90, fontsize=8)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Visit")
            ax.set_ylabel("Sky Under/Over-Subtraction % of Fibers")
            ax.set_ylim(ylim_lo_box, ylim_hi_box)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_title(f"PFSMerged SKY Under/Over-Subtraction % — All Arms\n"
                         f"Collections = {collections}{page_label}")
            ax.legend(handles=legend_elements, loc='upper right',
                      fancybox=True, framealpha=0.5, fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.minorticks_on()
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

        # --- Line plots: 3 arm median lines overlaid with ±1σ bands ---
        for page_idx, idx_chunk in enumerate(chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_pages})" if n_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
            for arm in ARMS:
                chunk_med = [medians[arm][i] for i in idx_chunk]
                chunk_p16 = [p16s[arm][i]    for i in idx_chunk]
                chunk_p84 = [p84s[arm][i]    for i in idx_chunk]
                ax.fill_between(range(n), chunk_p16, chunk_p84,
                                color=ARM_COLORS[arm], alpha=0.15)
                ax.plot(range(n), chunk_med, marker='o', color=ARM_COLORS[arm],
                        linewidth=1.2, markersize=4, label=ARM_LABELS[arm])
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90)
            ax.set_xlabel('Visit')
            ax.set_ylabel("Sky Under/Over-Subtraction % (Median per Visit)")
            ax.set_ylim(ylim_lo_line, ylim_hi_line)
            ax.set_title(f"PFSMerged SKY Under/Over-Subtraction % per Visit — All Arms\n"
                         f"Collections = {collections}{page_label}")
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='±1σ (p16–p84)'))
            ax.legend(handles=handles, loc='upper left', fancybox=True, framealpha=0.5)
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

        # --- Per-visit histograms: 3 arms overlaid, dynamic x-limits ---
        for visit in visits_order:
            df_v = df[df["Visit"] == visit]
            arm_data_visit = {arm: df_v[f"SkyLineResidual_pct_{arm}"].dropna().values for arm in ARMS}
            if all(len(d) == 0 for d in arm_data_visit.values()):
                continue

            all_vals = np.concatenate([d for d in arm_data_visit.values() if len(d) > 0])
            x_abs = min(10.0, max(abs(np.nanpercentile(all_vals, 2)),
                                  abs(np.nanpercentile(all_vals, 98))) * 1.1)
            x_abs = max(x_abs, 1.0)
            shared_bins = np.arange(-x_abs, x_abs + 0.25, 0.25)

            fig, ax = plt.subplots(figsize=(8, 5))
            for arm in ARMS:
                data = arm_data_visit[arm]
                if len(data) == 0:
                    continue
                ax.hist(data, bins=shared_bins, color=ARM_COLORS[arm], alpha=0.4,
                        edgecolor=ARM_COLORS[arm], linewidth=0.5,
                        weights=np.ones(len(data)) / len(data) * 100,
                        label=ARM_LABELS[arm])
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Sky Over/Under-Subtraction %")
            ax.set_ylabel("% of Fibers")
            ax.set_xlim(-x_abs, x_abs)
            ax.set_title(f"PFSMerged SKY Under/Over-Subtraction % — Visit {visit}\n"
                         f"Collections = {collections}")
            ax.legend(loc='upper left', fancybox=True, framealpha=0.5, fontsize=8)
            ax.grid(True, alpha=0.4)
            ax.minorticks_on()
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

    visits_no_data = [v for v in visits_order if v not in valid_visits]
    print(f"\n{'='*60}")
    print(f"  Sky Line QA Plots")
    print(f"{'='*60}")
    print(f"  Total visits            : {len(visits_order)}")
    print(f"  Visits plotted          : {n_valid}")
    print(f"  Visits skipped (no data): {len(visits_no_data)}  {visits_no_data if visits_no_data else ''}")
    print(f"{'='*60}")
    print(f"  Saved: {pdf_name}")
    print(f"{'='*60}")


# ==== CONTINUUM PLOTS: BOX PLOTS AND LINE PLOTS PER ARM ====
def plot_continuum_plots(continuum_file, collections, collections_safe, visits_order):
    visit_str = f"{visits_order[0]}-{visits_order[-1]}"
    pdf_name  = f"PFS_SkySubQA_ContinuumPlots_Visits={visit_str}_Collections={collections_safe}.pdf"

    def _whisker_bounds(data_lists):
        lows, highs = [], []
        for d in data_lists:
            d_c = d[np.isfinite(d)]
            if len(d_c) < 2:
                continue
            q1, q3 = np.percentile(d_c, [25, 75])
            iqr = q3 - q1
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            w_lo = float(d_c[d_c >= lo_fence].min()) if (d_c >= lo_fence).any() else float(q1)
            w_hi = float(d_c[d_c <= hi_fence].max()) if (d_c <= hi_fence).any() else float(q3)
            lows.append(w_lo)
            highs.append(w_hi)
        return lows, highs

    with PdfPages(pdf_name) as pdf:
        df_cont = pd.read_csv(continuum_file)

        all_data_flux  = {arm: [] for arm in ARMS}
        all_data_ratio = {arm: [] for arm in ARMS}
        valid_visits   = []

        for visit in visits_order:
            df_v = df_cont[df_cont["Visit"] == visit]
            if len(df_v) == 0:
                continue
            valid_visits.append(visit)
            for arm in ARMS:
                flux_col = f"FluxContinuum_{arm}"
                sky_col  = f"SkyContinuum_{arm}"
                df_valid = df_v[[flux_col, sky_col]].dropna()
                f_arr    = df_valid[flux_col].to_numpy()
                s_arr    = df_valid[sky_col].to_numpy()
                all_data_flux[arm].append(f_arr if len(f_arr) > 0 else np.array([np.nan]))
                if len(f_arr) > 0:
                    ratio = (f_arr / s_arr) * 100
                    ratio_clean = ratio[np.isfinite(ratio)]
                    all_data_ratio[arm].append(
                        ratio_clean if len(ratio_clean) > 0 else np.array([np.nan]))
                else:
                    all_data_ratio[arm].append(np.array([np.nan]))

        n_valid = len(valid_visits)

        # Global y-limits: flux box
        all_wlo_flux, all_whi_flux = [], []
        for arm in ARMS:
            lo, hi = _whisker_bounds(all_data_flux[arm])
            all_wlo_flux.extend(lo); all_whi_flux.extend(hi)
        if all_wlo_flux and all_whi_flux:
            pad_f = (np.percentile(all_whi_flux, 98) - np.percentile(all_wlo_flux, 2)) * 0.1
            ylim_lo_flux_box = np.percentile(all_wlo_flux, 2) - pad_f
            ylim_hi_flux_box = np.percentile(all_whi_flux, 98) + pad_f
        else:
            ylim_lo_flux_box, ylim_hi_flux_box = -100, 100

        # Global y-limits: ratio box
        all_wlo_ratio, all_whi_ratio = [], []
        for arm in ARMS:
            lo, hi = _whisker_bounds(all_data_ratio[arm])
            all_wlo_ratio.extend(lo); all_whi_ratio.extend(hi)
        if all_wlo_ratio and all_whi_ratio:
            pad_r = (np.percentile(all_whi_ratio, 98) - np.percentile(all_wlo_ratio, 2)) * 0.1
            ylim_lo_ratio_box = np.percentile(all_wlo_ratio, 2) - pad_r
            ylim_hi_ratio_box = np.percentile(all_whi_ratio, 98) + pad_r
        else:
            ylim_lo_ratio_box, ylim_hi_ratio_box = -10, 10

        # Per-arm per-visit stats for line plots
        flux_med  = {arm: [float(np.nanmedian(d))         for d in all_data_flux[arm]]  for arm in ARMS}
        flux_p16  = {arm: [float(np.nanpercentile(d, 16)) for d in all_data_flux[arm]]  for arm in ARMS}
        flux_p84  = {arm: [float(np.nanpercentile(d, 84)) for d in all_data_flux[arm]]  for arm in ARMS}
        ratio_med = {arm: [float(np.nanmedian(d))         for d in all_data_ratio[arm]] for arm in ARMS}
        ratio_p16 = {arm: [float(np.nanpercentile(d, 16)) for d in all_data_ratio[arm]] for arm in ARMS}
        ratio_p84 = {arm: [float(np.nanpercentile(d, 84)) for d in all_data_ratio[arm]] for arm in ARMS}

        # Global y-limits: flux line
        all_fp16 = [v for arm in ARMS for v in flux_p16[arm] if np.isfinite(v)]
        all_fp84 = [v for arm in ARMS for v in flux_p84[arm] if np.isfinite(v)]
        if all_fp16 and all_fp84:
            pad_fl = (np.percentile(all_fp84, 98) - np.percentile(all_fp16, 2)) * 0.15
            ylim_lo_flux_line = np.percentile(all_fp16, 2)  - pad_fl
            ylim_hi_flux_line = np.percentile(all_fp84, 98) + pad_fl
        else:
            ylim_lo_flux_line, ylim_hi_flux_line = -100, 100

        # Global y-limits: ratio line
        all_rp16 = [v for arm in ARMS for v in ratio_p16[arm] if np.isfinite(v)]
        all_rp84 = [v for arm in ARMS for v in ratio_p84[arm] if np.isfinite(v)]
        if all_rp16 and all_rp84:
            pad_rl = (np.percentile(all_rp84, 98) - np.percentile(all_rp16, 2)) * 0.15
            ylim_lo_ratio_line = np.percentile(all_rp16, 2)  - pad_rl
            ylim_hi_ratio_line = np.percentile(all_rp84, 98) + pad_rl
        else:
            ylim_lo_ratio_line, ylim_hi_ratio_line = -10, 10

        box_chunks  = [list(range(i, min(i + VISITS_PER_PAGE, n_valid)))
                       for i in range(0, n_valid, VISITS_PER_PAGE)]
        n_box_pages = len(box_chunks)

        legend_elements = [
            Patch(facecolor=ARM_COLORS[arm], edgecolor=ARM_COLORS[arm], alpha=0.7, label=ARM_LABELS[arm])
            for arm in ARMS
        ]

        # ---- Flux continuum box plots (3 arms side by side) ----
        for page_idx, idx_chunk in enumerate(box_chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_box_pages})" if n_box_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(10, n * 1.0), 6))
            for arm in ARMS:
                positions  = [i + ARM_OFFSETS[arm] for i in range(n)]
                chunk_data = [all_data_flux[arm][i] for i in idx_chunk]
                box = ax.boxplot(chunk_data, positions=positions, widths=BOX_WIDTH,
                                 patch_artist=True,
                                 medianprops=dict(color='black', linewidth=1.5))
                for patch in box['boxes']:
                    patch.set_facecolor(ARM_COLORS[arm])
                    patch.set_alpha(0.6)
                    patch.set_edgecolor(ARM_COLORS[arm])
                for w in box['whiskers']: w.set_color(ARM_COLORS[arm])
                for c in box['caps']:     c.set_color(ARM_COLORS[arm])
                for f in box['fliers']:   f.set_markeredgecolor(ARM_COLORS[arm]); f.set_markersize(2)

            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90, fontsize=8)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Visit")
            ax.set_ylabel("Median Flux Continuum of SKY Fibers [e-/nm]")
            ax.set_ylim(ylim_lo_flux_box, ylim_hi_flux_box)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_title(f"PFSMerged SKY Fiber Flux Continuum per Visit — ALL Arms\n"
                         f"Collections={collections}{page_label}")
            ax.legend(handles=legend_elements, loc='upper right',
                      fancybox=True, framealpha=0.5, fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.minorticks_on()
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

        # ---- Flux continuum line plots (3 arm lines overlaid) ----
        for page_idx, idx_chunk in enumerate(box_chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_box_pages})" if n_box_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
            for arm in ARMS:
                chunk_med = [flux_med[arm][i] for i in idx_chunk]
                chunk_p16 = [flux_p16[arm][i] for i in idx_chunk]
                chunk_p84 = [flux_p84[arm][i] for i in idx_chunk]
                ax.fill_between(range(n), chunk_p16, chunk_p84,
                                color=ARM_COLORS[arm], alpha=0.15)
                ax.plot(range(n), chunk_med, marker='o', color=ARM_COLORS[arm],
                        linewidth=1.2, markersize=4, label=ARM_LABELS[arm])
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90)
            ax.set_xlabel('Visit')
            ax.set_ylabel("Median Flux Continuum of SKY Fibers [e-/nm]")
            ax.set_ylim(ylim_lo_flux_line, ylim_hi_flux_line)
            ax.set_title(f"PFSMerged SKY Flux Continuum per Visit — ALL Arms\n"
                         f"Collections={collections}{page_label}")
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='±1σ (p16–p84)'))
            ax.legend(handles=handles, loc='upper left', fancybox=True, framealpha=0.5)
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

        # ---- Flux/Sky ratio box plots (3 arms side by side) ----
        for page_idx, idx_chunk in enumerate(box_chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_box_pages})" if n_box_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(10, n * 1.0), 6))
            for arm in ARMS:
                positions  = [i + ARM_OFFSETS[arm] for i in range(n)]
                chunk_data = [all_data_ratio[arm][i] for i in idx_chunk]
                box = ax.boxplot(chunk_data, positions=positions, widths=BOX_WIDTH,
                                 patch_artist=True,
                                 medianprops=dict(color='black', linewidth=1.5))
                for patch in box['boxes']:
                    patch.set_facecolor(ARM_COLORS[arm])
                    patch.set_alpha(0.6)
                    patch.set_edgecolor(ARM_COLORS[arm])
                for w in box['whiskers']: w.set_color(ARM_COLORS[arm])
                for c in box['caps']:     c.set_color(ARM_COLORS[arm])
                for f in box['fliers']:   f.set_markeredgecolor(ARM_COLORS[arm]); f.set_markersize(2)

            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90, fontsize=8)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Visit")
            ax.set_ylabel("Flux/Sky Continuum Ratio of SKY Fibers [%]")
            ax.set_ylim(ylim_lo_ratio_box, ylim_hi_ratio_box)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_title(f"PFSMerged Flux/Sky Continuum Ratio per Visit — ALL Arms\n"
                         f"Collections={collections}{page_label}")
            ax.legend(handles=legend_elements, loc='upper right',
                      fancybox=True, framealpha=0.5, fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.minorticks_on()
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

        # ---- Flux/Sky ratio line plots (3 arm lines overlaid) ----
        for page_idx, idx_chunk in enumerate(box_chunks):
            chunk_visits = [valid_visits[i] for i in idx_chunk]
            n = len(chunk_visits)
            page_label = f" (Page {page_idx + 1}/{n_box_pages})" if n_box_pages > 1 else ""

            fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
            for arm in ARMS:
                chunk_med = [ratio_med[arm][i] for i in idx_chunk]
                chunk_p16 = [ratio_p16[arm][i] for i in idx_chunk]
                chunk_p84 = [ratio_p84[arm][i] for i in idx_chunk]
                ax.fill_between(range(n), chunk_p16, chunk_p84,
                                color=ARM_COLORS[arm], alpha=0.15)
                ax.plot(range(n), chunk_med, marker='o', color=ARM_COLORS[arm],
                        linewidth=1.2, markersize=4, label=ARM_LABELS[arm])
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xticks(range(n))
            ax.set_xticklabels(chunk_visits, rotation=90)
            ax.set_xlabel('Visit')
            ax.set_ylabel("Flux/Sky Continuum Ratio of SKY Fibers [%]")
            ax.set_ylim(ylim_lo_ratio_line, ylim_hi_ratio_line)
            ax.set_title(f"PFSMerged Flux/Sky Continuum Ratio per Visit — ALL Arms\n"
                         f"Collections={collections}{page_label}")
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor='gray', alpha=0.3, edgecolor='none', label='±1σ (p16–p84)'))
            ax.legend(handles=handles, loc='upper left', fancybox=True, framealpha=0.5)
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            _save_raster(pdf)
            plt.close()

    visits_no_data = [v for v in visits_order if v not in valid_visits]
    print(f"\n{'='*60}")
    print(f"  Continuum Fit Plots (per arm)")
    print(f"{'='*60}")
    print(f"  Total visits            : {len(visits_order)}")
    print(f"  Visits plotted          : {n_valid}")
    print(f"  Visits skipped (no data): {len(visits_no_data)}  {visits_no_data if visits_no_data else ''}")
    print(f"{'='*60}")
    print(f"  Saved: {pdf_name}")
    print(f"{'='*60}")


# ==== HELPER: PLOT MEDIAN SPECTRA (4-PANEL LAYOUT) FOR A LIST OF VISITS ====
def _plot_median_spectra(visit_list, collections, df_skyline, repo, pdf, label, refLines, task):
    visits_ok, visits_nodata, visits_failed = [], [], []

    for visit in visit_list:
        print(f"\n=== [{label}] Processing Visit {visit} ===")

        df_visit_fibers = df_skyline[
            (df_skyline["Visit"] == visit) &
            (df_skyline[[f"SkyLineResidual_pct_{arm}" for arm in ARMS]].notna().any(axis=1))
        ].copy()

        if len(df_visit_fibers) == 0:
            print(f"  [SKIP] Visit {visit}: no SKY fibers with sky line data found.")
            visits_nodata.append(visit)
            continue

        try:
            butler = Butler(repo, collections=collections)
            pfsMerged_visit = butler.get('pfsMerged', visit=visit)

            fiberid_all = df_visit_fibers["Fiberid"].to_numpy()

            # Use a non-spectrograph-2 fiber as the wavelength reference and template
            ref_pfsMerged = None
            for fid in fiberid_all:
                candidate = pfsMerged_visit.select(fiberId=fid)
                if candidate.spectrograph[0] != 2:
                    ref_pfsMerged = candidate
                    break
            if ref_pfsMerged is None:
                ref_pfsMerged = pfsMerged_visit.select(fiberId=fiberid_all[0])
            wav_common = ref_pfsMerged.wavelength[0]

            # Stack fluxes with uniform quality masking: any pixel flagged bad or
            # with zero/invalid norm is excluded for every fiber, regardless of
            # spectrograph or arm — bad data is dropped by matching the data itself.
            flux_arrays = []
            norm_arrays = []
            for fid in fiberid_all:
                try:
                    pfs_fiber  = pfsMerged_visit.select(fiberId=fid)
                    pixels_bad = (pfs_fiber.mask & pfs_fiber.flags.get(
                        'BAD', 'CR', 'SAT', 'NO_DATA')) != 0
                    flux = pfs_fiber.flux[0].copy().astype(np.float64)
                    norm = pfs_fiber.norm[0].copy().astype(np.float64)
                    norm_zero       = ~np.isfinite(norm) | (np.abs(norm) < 1e-10)
                    effectively_bad = pixels_bad[0] | norm_zero
                    flux[effectively_bad] = np.nan
                    norm[effectively_bad] = np.nan
                    flux_arrays.append(flux)
                    norm_arrays.append(norm)
                except Exception as e:
                    print(f"    Visit {visit} fiber {fid} skipped: {e}")
                    continue

            if not flux_arrays:
                visits_nodata.append(visit)
                continue

            flux_stack = np.vstack(flux_arrays)
            norm_stack = np.vstack(norm_arrays)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median_flux = np.nanmedian(flux_stack, axis=0)
                median_norm = np.nanmedian(norm_stack, axis=0).astype(np.float32)
                p16_flux    = np.nanpercentile(flux_stack, 16, axis=0)
                p84_flux    = np.nanpercentile(flux_stack, 84, axis=0)
                p2_flux     = np.nanpercentile(flux_stack,  2, axis=0)
                p98_flux    = np.nanpercentile(flux_stack, 98, axis=0)

            # Fit continuum on median spectrum
            nan_mask = np.isnan(median_flux)
            median_flux_for_fit = np.where(nan_mask, 0.0, median_flux).astype(np.float32)
            try:
                spectra_med = SpectrumSet.fromPfsArm(ref_pfsMerged)
                for s in spectra_med:
                    s.setFlux(np.ascontiguousarray(median_flux_for_fit, dtype=np.float32))
                    s.norm = np.ascontiguousarray(median_norm, dtype=np.float32)
                    new_mask = s.mask.array.copy()
                    new_mask[0, nan_mask] = ref_pfsMerged.flags['NO_DATA']
                    s.mask.array[:] = new_mask
                cont_fit    = task.run(spectra_med, refLines)
                cont_median = (cont_fit * median_norm[np.newaxis, :])[0].copy().astype(np.float64)
                cont_median[nan_mask] = np.nan
            except Exception:
                cont_median = np.full_like(median_flux, np.nan)
            plt.close("all")

            # Y-limits: full spectrum top panel
            band_lo  = np.nanpercentile(p16_flux, 20)
            band_hi  = np.nanpercentile(p84_flux, 80)
            pad_full = (band_hi - band_lo) * 0.10
            ylim_full = (band_lo - pad_full, band_hi + pad_full)

            # Per-arm y-limits from clean extraction range, then share the widest across all arms
            arm_ylims = {}
            for arm in ARMS:
                lo, hi = CONT_RANGES[arm]
                mask_a = (wav_common >= lo) & (wav_common <= hi)
                flux_a = median_flux[mask_a]
                plo, phi = ARM_YLIM_PCTS[arm]
                a_lo  = np.nanpercentile(flux_a, plo)
                a_hi  = np.nanpercentile(flux_a, phi)
                a_pad = (a_hi - a_lo) * 0.10
                arm_ylims[arm] = (a_lo - a_pad, a_hi + a_pad)
            valid_bounds = [(lo, hi) for lo, hi in arm_ylims.values()
                            if np.isfinite(lo) and np.isfinite(hi)]
            if valid_bounds:
                shared_lo = min(v[0] for v in valid_bounds)
                shared_hi = max(v[1] for v in valid_bounds)
            else:
                shared_lo, shared_hi = -10, 10

            # 4-panel figure: full spectrum on top, 3 arm zoom panels on bottom
            fig = plt.figure(figsize=(16, 12))
            gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.3)
            ax_top = fig.add_subplot(gs[0, :])
            ax_arm = {'b': fig.add_subplot(gs[1, 0]),
                      'r': fig.add_subplot(gs[1, 1]),
                      'n': fig.add_subplot(gs[1, 2])}

            ax_top.fill_between(wav_common, p2_flux, p98_flux,
                                color='gray', alpha=0.15, label='2nd–98th %ile')
            ax_top.fill_between(wav_common, p16_flux, p84_flux,
                                color='gray', alpha=0.30, label='16th–84th %ile')
            ax_top.plot(wav_common, median_flux, color='blue',   linewidth=0.5,
                        label=f'Median flux  (N={len(flux_arrays)})')
            ax_top.plot(wav_common, cont_median, color='orange', linewidth=1.2,
                        label='Continuum fit')
            ax_top.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
            ax_top.set_xlim(float(wav_common.min()), float(wav_common.max()))
            ax_top.set_ylim(*ylim_full)
            ax_top.set_xlabel("Wavelength [nm]")
            ax_top.set_ylabel("Flux [e-/nm]")
            ax_top.set_title(
                f"PFSMerged ALL SKY Fibers Median Spectrum  Visit={visit}  [{label}]\n"
                f"Collections={collections}")
            ax_top.legend(fontsize=8, loc='upper left', fancybox=True, framealpha=0.5)
            ax_top.minorticks_on()
            ax_top.grid(True, alpha=0.3)

            for arm, ax_a in ax_arm.items():
                flo, fhi = ARM_FULL_RANGES[arm]
                mask_a = (wav_common >= flo) & (wav_common <= fhi)
                wav_a  = wav_common[mask_a]
                flux_a = median_flux[mask_a]
                cont_a = cont_median[mask_a]

                if len(wav_a) == 0:
                    ax_a.set_visible(False)
                    continue

                ax_a.plot(wav_a, flux_a, color=ARM_COLORS[arm], linewidth=0.5, label='Median flux')
                ax_a.plot(wav_a, cont_a, color='black',          linewidth=1.2, label='Continuum fit')
                ax_a.axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
                ax_a.set_xlim(flo, fhi)
                ax_a.set_ylim(shared_lo, shared_hi)
                ax_a.set_xlabel("Wavelength [nm]")
                ax_a.set_ylabel("Flux [e-/nm]")
                ax_a.set_title(ARM_LABELS[arm])
                ax_a.legend(fontsize=7, loc='upper left', fancybox=True, framealpha=0.5)
                ax_a.minorticks_on()
                ax_a.grid(True, alpha=0.3)

            fig.tight_layout()
            _save_raster(pdf)
            plt.close()
            gc.collect()

            print(f"  [OK]   Visit {visit}: plotted successfully ({len(flux_arrays)} SKY fibers).")
            visits_ok.append(visit)

        except Exception as e:
            plt.close('all')
            print(f"  [FAIL] Visit {visit}: {e}")
            visits_failed.append(visit)

    print(f"\n{'='*55}")
    print(f"  Median Spectra Plot Summary [{label}]")
    print(f"{'='*55}")
    print(f"  Total visits        : {len(visit_list)}")
    print(f"  Successfully plotted: {len(visits_ok)}")
    print(f"  Skipped (no data)   : {len(visits_nodata)}  {visits_nodata if visits_nodata else ''}")
    print(f"  Failed (error)      : {len(visits_failed)}  {visits_failed if visits_failed else ''}")
    print(f"{'='*55}")


# ==== MEDIAN SPECTRA FOR SAMPLED VISITS: BEST / WORST-SPREAD / WORST-OFFSET ====
def plot_pfs_merged_spectra(collections, collections_safe, visits_order, skyline_file, summary_file, repo):
    N_EACH = 10

    df_s = pd.read_csv(summary_file)
    df_s = df_s[df_s["Visit"].isin(visits_order)].copy()
    df_s["_score"] = (df_s["ContinuumFlux_median_abs"].fillna(0) +
                      df_s["ContinuumFlux_spread"].fillna(0))

    visit_str = f"{visits_order[0]}-{visits_order[-1]}"
    pdf_name  = f"PFS_SkySubQA_MedianSpectra_Visits={visit_str}_Collections={collections_safe}.pdf"

    col_abs    = "ContinuumFlux_median_abs"
    col_spread = "ContinuumFlux_spread"

    # Worst by absolute continuum offset
    worst_abs_visits = (df_s.nlargest(min(N_EACH, len(df_s)), col_abs)
                           .sort_values(col_abs, ascending=False)["Visit"].tolist())
    seen = set(worst_abs_visits)

    # Worst by continuum spread, skipping already selected
    worst_spread_visits = [v for v in df_s.sort_values(col_spread, ascending=False)["Visit"].tolist()
                           if v not in seen][:N_EACH]
    seen |= set(worst_spread_visits)

    # Best overall (lowest combined score), skipping already selected.
    # Only rank visits that actually have continuum data — a visit with NaN in both
    # metrics would otherwise get _score=0 (via fillna) and falsely rank as "best".
    good_pool   = df_s.dropna(subset=[col_abs, col_spread], how="all")
    good_visits = [v for v in good_pool.sort_values("_score", ascending=True)["Visit"].tolist()
                   if v not in seen][:N_EACH]

    n_total_sampled = len(good_visits) + len(worst_spread_visits) + len(worst_abs_visits)
    print(f"\n  Median spectra: {len(visits_order)} total visits, plotting {n_total_sampled} sampled")
    print(f"    Best overall : {len(good_visits)}  |  Worst spread : {len(worst_spread_visits)}"
          f"  |  Worst offset : {len(worst_abs_visits)}")

    # Compute once and share across all three visit sections
    _config = ReadLineListTask.ConfigClass()
    _config.lightSources = ["sky"]
    _config.lampElements = []
    _config.exclusionRadius = 1.0
    refLines = ReadLineListTask(_config).run()
    refLines = refLines[(refLines.wavelength > 350) &
                        (refLines.wavelength < 1300) &
                        (refLines.intensity > 100) &
                        (refLines.status == 0)]
    task       = FitContinuumTask(FitContinuumTask.ConfigClass())
    df_skyline = pd.read_csv(skyline_file)

    def _section_page(pdf, title, color):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.text(0.5, 0.55, title, transform=ax.transAxes,
                fontsize=28, fontweight='bold', ha='center', va='center', color=color)
        _save_raster(pdf)
        plt.close()

    with PdfPages(pdf_name) as pdf:
        _section_page(pdf, "Best Visits", color='#2166ac')
        _plot_median_spectra(good_visits, collections, df_skyline,
                             repo, pdf, label="BEST",
                             refLines=refLines, task=task)

        _section_page(pdf, "Worst Visits by Spread", color='#d6604d')
        _plot_median_spectra(worst_spread_visits, collections, df_skyline,
                             repo, pdf, label="WORST-SPREAD",
                             refLines=refLines, task=task)

        _section_page(pdf, "Worst Visits by Offset", color='#b2182b')
        _plot_median_spectra(worst_abs_visits, collections, df_skyline,
                             repo, pdf, label="WORST-OFFSET",
                             refLines=refLines, task=task)

    print(f"{'='*55}")
    print(f"  Saved: {pdf_name}")
    print(f"{'='*55}")


# ==== EXECUTION ====
if __name__ == "__main__":
    # repo        = "/lfs_pfs/Subaru/PFS/data/datastore_20260206"
    # collections = "PFS/science/S25A/coadd.20260206"
    
    repo        = "/lfs_pfs/Subaru/PFS/data/datastore_20260226"
    # collections = "PFS/science/run26/coadd.20260430"
    # collections = "PFS/science/run28/coadd.20260707/brn"
    collections = "PFS/science/S25/coadd.20260804"

    cpu_cores = 50

    butler           = Butler(repo, collections=collections)
    collections_safe = collections.replace('/', '_')
    visits           = sorted({dataRef.dataId['visit']
                                for dataRef in butler.registry.queryDatasets('pfsMerged')})
    #visits = sorted([141567, 141568])

    visit_str = f"{visits[0]}-{visits[-1]}"
    log_file  = f"SkySubQA_Log_Visits={visit_str}_Collections={collections_safe}.txt"

    with _Tee(log_file):
        df_summary, skyline_file, summary_file, continuum_file, visits_order, collections_safe = \
            process_visits(visits, collections, cpu_cores, repo)
        plot_skysub(skyline_file, collections, collections_safe, visits_order)
        plot_continuum_plots(continuum_file, collections, collections_safe, visits_order)
        plot_pfs_merged_spectra(collections, collections_safe, visits_order,
                                skyline_file, summary_file, repo)
