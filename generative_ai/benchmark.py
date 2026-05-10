import argparse
import csv
import os
import time
import multiprocessing as mp
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.modelv11 import postprocess_image, mdn_point_and_confidence
from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import smooth_scale, inv_smooth_scale
from src.visualizer import plot_line_evolution


# ============================================================
# CONFIG
# ============================================================
MODEL_ROOT = "results/good_v2"
CKPT_PATH = os.path.join(MODEL_ROOT, "ckpt", "best.pt")
OUT_DIR = os.path.join(MODEL_ROOT, "sim_time_benchmark_parallel4")

IMAGE_SIZE = (48, 48)
STRICT = False
VAR_SCALE = 1

# 4 核 CPU 并行时建议强制 CPU，避免多个进程抢同一块 GPU
FORCE_CPU = True


# ============================================================
# Basic utils
# ============================================================
def get_device():
    if FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def get_init_tensor(image_size: tuple):
    base_eta = np.zeros((50, 50), dtype=np.float32)
    base_eta[:, :5] = 1.0

    base_c = np.zeros((50, 50), dtype=np.float32)
    base_c[:, :5] = 0.2
    base_c[:, 5:] = 0.8

    base_p = np.zeros((50, 50), dtype=np.float32)

    arr = np.stack([base_eta, base_c, base_p], axis=-1)
    arr = cv2.resize(arr, image_size, interpolation=cv2.INTER_LINEAR)

    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)

    return tensor_t


def get_sim_range_s(t_values):
    if len(t_values) <= 1:
        return 0.0
    return float(max(t_values) - min(t_values))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# Plot / save helpers
# ============================================================
def plot_latent_exploration(
    run_dir,
    z_path,
    scores=None,
    coverages=None,
    cand_clouds=None,
    cand_values=None,
    value_name="H",
    colorize_candidates=False,
    show_step_labels=True,
    max_step_labels=30,
):
    os.makedirs(run_dir, exist_ok=True)

    Zpath = np.asarray(z_path)
    T_plus_1, D = Zpath.shape
    T = T_plus_1 - 1

    if cand_clouds is not None and len(cand_clouds) != T:
        raise ValueError(f"cand_clouds length must be T={T}, got {len(cand_clouds)}")

    if colorize_candidates:
        if cand_values is None or len(cand_values) != T:
            raise ValueError("colorize_candidates=True requires cand_values with length T")

    if cand_clouds is None:
        Z_all = Zpath
    else:
        Z_all = [Zpath]
        for C in cand_clouds:
            Z_all.append(np.asarray(C))
        Z_all = np.concatenate(Z_all, axis=0)

    mean = Z_all.mean(axis=0)
    Zc = Z_all - mean
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    W = Vt[:2].T

    Zp2 = (Zpath - mean) @ W

    plt.figure(figsize=(7.5, 6.5))
    mappable = None

    if cand_clouds is not None:
        if colorize_candidates:
            vmin = min(float(np.min(v)) for v in cand_values)
            vmax = max(float(np.max(v)) for v in cand_values)

        for t, C in enumerate(cand_clouds):
            C = np.asarray(C)
            C2 = (C - mean) @ W

            if colorize_candidates:
                vals = np.asarray(cand_values[t])
                sc = plt.scatter(
                    C2[:, 0],
                    C2[:, 1],
                    c=vals,
                    vmin=vmin,
                    vmax=vmax,
                    s=10,
                    alpha=0.25,
                    linewidths=0,
                )
                mappable = sc
            else:
                plt.scatter(
                    C2[:, 0],
                    C2[:, 1],
                    s=10,
                    alpha=0.18,
                    color="gray",
                    linewidths=0,
                )

            z_best = Zpath[t + 1]
            z_best2 = (z_best - mean) @ W
            plt.scatter(
                z_best2[0],
                z_best2[1],
                s=90,
                marker="*",
                color="gold",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

    plt.plot(
        Zp2[:, 0],
        Zp2[:, 1],
        "-o",
        linewidth=1.6,
        markersize=4,
        label="accepted path",
        zorder=4,
    )

    for i in range(len(Zp2) - 1):
        plt.annotate(
            "",
            xy=(Zp2[i + 1, 0], Zp2[i + 1, 1]),
            xytext=(Zp2[i, 0], Zp2[i, 1]),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    if show_step_labels:
        stride = max(1, T_plus_1 // max_step_labels)
        for i in range(0, T_plus_1, stride):
            plt.text(Zp2[i, 0], Zp2[i, 1], str(i), fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if mappable is not None:
        cb = plt.colorbar(mappable, fraction=0.046, pad=0.04)
        cb.set_label(value_name)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "latent_exploration_pca2d.png"), dpi=220)
    plt.close()

    steps = np.arange(T_plus_1)

    plot_line_evolution(
        steps,
        np.linalg.norm(Zpath, axis=1),
        xlabel="step",
        ylabel="||z||",
        save_path=os.path.join(run_dir, "latent_norm_over_steps.png"),
    )

    if scores is not None:
        plot_line_evolution(
            steps,
            scores,
            xlabel="step",
            ylabel="score",
            save_path=os.path.join(run_dir, "score_over_steps.png"),
        )

    if coverages is not None:
        plot_line_evolution(
            steps,
            coverages,
            xlabel="step",
            ylabel="dendrite coverage",
            save_path=os.path.join(run_dir, "coverage_over_steps.png"),
        )


def save_step(out_dir, step, img, z, params, coverage, score, hopping_strength):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="coolwarm")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"step_{step:03d}.png"), dpi=200)
    plt.close()

    csv_path = os.path.join(out_dir, "params_recording.csv")
    header = [
        "step",
        "score",
        "coverage",
        "hopping_strength",
        "t",
        "POT_LEFT",
        "fo",
        "Al",
        "Bl",
        "Cl",
        "As",
        "Bs",
        "Cs",
        "cleq",
        "cseq",
        "L1o",
        "L2o",
        "ko",
        "Noise",
    ]

    params = np.asarray(params).reshape(-1)
    if params.size != 15:
        raise ValueError(f"Expected params length=15, got {params.size}: {params}")

    row = [
        int(step),
        float(score),
        float(coverage),
        float(hopping_strength),
        *[float(x) for x in params.tolist()],
    ]

    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ============================================================
# One benchmark run
# ============================================================
def explore_until_sim_time(
    model,
    device,
    target_sim_range_s,
    rw_sigma,
    num_cand,
    run_id,
    max_steps,
    save_figures=True,
):
    run_dir = os.path.join(
        OUT_DIR,
        f"target_{target_sim_range_s:g}s_sigma_{rw_sigma:g}_cand_{num_cand}_run_{run_id}_{time.time()}",
    )
    os.makedirs(run_dir, exist_ok=True)

    timing_records = []

    explore_t0 = sync_time(device)

    # initial reconstruction
    recon_t0 = sync_time(device)

    with torch.no_grad():
        recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(
            get_init_tensor(IMAGE_SIZE).unsqueeze(0).to(device)
        )

        theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
            pi_s,
            mu_s,
            log_sigma_s,
            var_scale=VAR_SCALE,
        )

    recon = inv_smooth_scale(recon)
    recon = recon.cpu().detach().numpy()[0, 0]
    recon = postprocess_image(recon)

    z = z.cpu().detach().numpy()[0]
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]

    recon_elapsed = sync_time(device) - recon_t0
    timing_records.append({
        "section": "single_image_reconstruction",
        "elapsed_s": recon_elapsed,
        "target_sim_range_s": target_sim_range_s,
        "rw_sigma": rw_sigma,
        "num_cand": num_cand,
        "run_id": run_id,
        "step": 0,
        "kind": "initial",
        "run_dir": run_dir,
    })

    _, metrics, scores = generate_analysis_figure(np.clip(recon, 0, 1))
    s = scores["empirical_score"]
    c = metrics["dendrite_coverage"]
    t = float(y_pred_s[0])

    save_step(run_dir, 0, recon, z, y_pred_s, c, s, hopping_strength=rw_sigma)

    z_path = [z.copy()]
    cand_clouds = []
    cand_H = []
    score_path = [float(s)]
    coverage_path = [float(c)]
    t_path = [float(t)]

    stop_reason = "max_steps_reached"

    for step in range(1, max_steps + 1):
        if get_sim_range_s(t_path) >= target_sim_range_s:
            stop_reason = "target_sim_range_reached"
            break

        best_z = None
        best_H_score = -1e18
        best_score = -1e18
        best_img = None
        best_params = None
        best_coverage = None

        z_cands = []
        H_list = []

        for cand_idx in range(num_cand):
            recon_t0 = sync_time(device)

            dz = np.random.randn(*z.shape).astype(np.float32) * rw_sigma
            z_cand = z + dz
            z_cand_tensor = torch.from_numpy(z_cand).unsqueeze(0).to(device)

            with torch.no_grad():
                recon_cand, (
                    theta_hat_s_cand,
                    conf_param_s_cand,
                    conf_global_s_cand,
                    modes_s_cand,
                ) = model.inference(z_cand_tensor, var_scale=VAR_SCALE)

            recon_cand = inv_smooth_scale(recon_cand)
            recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
            recon_cand = postprocess_image(recon_cand)

            y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]

            recon_elapsed = sync_time(device) - recon_t0
            timing_records.append({
                "section": "single_image_reconstruction",
                "elapsed_s": recon_elapsed,
                "target_sim_range_s": target_sim_range_s,
                "rw_sigma": rw_sigma,
                "num_cand": num_cand,
                "run_id": run_id,
                "step": step,
                "candidate_idx": cand_idx,
                "kind": "candidate",
                "run_dir": run_dir,
            })

            _, metrics_cand, scores_cand = generate_analysis_figure(np.clip(recon_cand, 0, 1))

            t_cand = float(y_pred_s_cand[0])
            s_cand = scores_cand["empirical_score"]
            cnn_cand = metrics_cand["connected_components"]
            c_cand = metrics_cand["dendrite_coverage"]

            H = -np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

            z_cands.append(z_cand.copy())
            H_list.append(float(H))

            if c_cand < c or t_cand < t or (cnn_cand >= 3 and STRICT):
                continue

            if H > best_H_score:
                best_H_score = H
                best_score = s_cand
                best_z = z_cand
                best_img = recon_cand
                best_params = y_pred_s_cand
                best_coverage = c_cand

        if best_z is None:
            stop_reason = "no_valid_candidate"
            break

        z = best_z
        s = best_score
        c = best_coverage
        t = float(best_params[0])
        y_pred_s = best_params

        save_step(
            run_dir,
            step,
            best_img,
            best_z,
            best_params,
            best_coverage,
            best_score,
            hopping_strength=rw_sigma,
        )

        z_path.append(z.copy())
        score_path.append(float(s))
        coverage_path.append(float(c))
        t_path.append(float(t))

        cand_clouds.append(np.stack(z_cands, axis=0))
        cand_H.append(np.array(H_list, dtype=float))

        if get_sim_range_s(t_path) >= target_sim_range_s:
            stop_reason = "target_sim_range_reached"
            break

    if save_figures:
        try:
            plot_latent_exploration(
                run_dir,
                z_path,
                scores=score_path,
                coverages=coverage_path,
                cand_clouds=cand_clouds,
                cand_values=cand_H,
                colorize_candidates=True,
            )
        except Exception as e:
            print(f"[Warning] failed to plot {run_dir}: {e}")

    explore_elapsed = sync_time(device) - explore_t0

    simulation_start_s = float(min(t_path))
    simulation_end_s = float(max(t_path))
    achieved_simulation_range_s = float(simulation_end_s - simulation_start_s)
    steps_completed = int(len(t_path) - 1)

    timing_records.append({
        "section": "exploration",
        "elapsed_s": explore_elapsed,
        "target_sim_range_s": target_sim_range_s,
        "achieved_simulation_start_s": simulation_start_s,
        "achieved_simulation_end_s": simulation_end_s,
        "achieved_simulation_range_s": achieved_simulation_range_s,
        "rw_sigma": rw_sigma,
        "num_cand": num_cand,
        "run_id": run_id,
        "steps_completed": steps_completed,
        "max_steps": max_steps,
        "stop_reason": stop_reason,
        "run_dir": run_dir,
    })

    # per-run files
    timing_path = os.path.join(run_dir, "timing_records.csv")
    write_dict_csv(timing_path, timing_records)

    summary_row = {
        "target_sim_range_s": target_sim_range_s,
        "achieved_simulation_start_s": simulation_start_s,
        "achieved_simulation_end_s": simulation_end_s,
        "achieved_simulation_range_s": achieved_simulation_range_s,
        "exploration_real_elapsed_s": explore_elapsed,
        "rw_sigma": rw_sigma,
        "num_cand": num_cand,
        "run_id": run_id,
        "steps_completed": steps_completed,
        "accepted_step_runtime_s": explore_elapsed / steps_completed if steps_completed > 0 else np.nan,
        "max_steps": max_steps,
        "stop_reason": stop_reason,
        "run_dir": run_dir,
    }

    return {
        "summary": summary_row,
        "timing_records": timing_records,
    }


# ============================================================
# Multiprocessing worker
# ============================================================
def worker_run(task):
    # 控制每个进程内部线程数，避免 4 个 worker 每个又开很多 MKL/OpenMP 线程导致过载
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (
        target_sim_range_s,
        rw_sigma,
        num_cand,
        run_id,
        max_steps,
        save_figures,
    ) = task

    device = get_device()

    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    np.random.seed(int(time.time() * 1000000) % 2**32 + int(run_id))
    torch.manual_seed(int(time.time() * 1000000) % 2**32 + int(run_id))

    print(
        f"[Start] target={target_sim_range_s}s, "
        f"sigma={rw_sigma}, cand={num_cand}, run={run_id}, device={device}"
    )

    result = explore_until_sim_time(
        model=model,
        device=device,
        target_sim_range_s=target_sim_range_s,
        rw_sigma=rw_sigma,
        num_cand=num_cand,
        run_id=run_id,
        max_steps=max_steps,
        save_figures=save_figures,
    )

    s = result["summary"]
    print(
        f"[Done] target={target_sim_range_s}s, "
        f"sigma={rw_sigma}, cand={num_cand}, run={run_id}, "
        f"stop={s['stop_reason']}, "
        f"range={s['achieved_simulation_range_s']:.4f}s, "
        f"time={s['exploration_real_elapsed_s']:.2f}s, "
        f"steps={s['steps_completed']}"
    )

    return result


# ============================================================
# CSV / summary
# ============================================================
def write_dict_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(run_rows, timing_records, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    run_path = os.path.join(out_dir, "sim_time_benchmark_runs.csv")
    timing_path = os.path.join(out_dir, "timing_records.csv")
    overall_path = os.path.join(out_dir, "direct_runtime_summary.csv")
    group_path = os.path.join(out_dir, "direct_runtime_group_summary.csv")

    write_dict_csv(run_path, run_rows)
    write_dict_csv(timing_path, timing_records)

    successful_rows = [
        r for r in run_rows
        if r["stop_reason"] == "target_sim_range_reached"
    ]

    recon_records = [
        r for r in timing_records
        if r["section"] == "single_image_reconstruction"
    ]

    overall_rows = []
    target_values = sorted({r["target_sim_range_s"] for r in run_rows})

    for target in target_values:
        rows_t = [
            r for r in successful_rows
            if float(r["target_sim_range_s"]) == float(target)
        ]

        recon_t = [
            r for r in recon_records
            if float(r["target_sim_range_s"]) == float(target)
        ]

        if rows_t:
            total_times = np.asarray([safe_float(r["exploration_real_elapsed_s"]) for r in rows_t], dtype=float)
            step_times = np.asarray([safe_float(r["accepted_step_runtime_s"]) for r in rows_t], dtype=float)
            achieved_ranges = np.asarray([safe_float(r["achieved_simulation_range_s"]) for r in rows_t], dtype=float)
            steps = np.asarray([safe_float(r["steps_completed"]) for r in rows_t], dtype=float)

            total_avg = float(np.nanmean(total_times))
            total_min = float(np.nanmin(total_times))
            total_max = float(np.nanmax(total_times))

            step_avg = float(np.nanmean(step_times))
            step_min = float(np.nanmin(step_times))
            step_max = float(np.nanmax(step_times))

            range_avg = float(np.nanmean(achieved_ranges))
            range_min = float(np.nanmin(achieved_ranges))
            range_max = float(np.nanmax(achieved_ranges))

            steps_avg = float(np.nanmean(steps))
        else:
            total_avg = total_min = total_max = np.nan
            step_avg = step_min = step_max = np.nan
            range_avg = range_min = range_max = np.nan
            steps_avg = np.nan

        if recon_t:
            recon_times = np.asarray([safe_float(r["elapsed_s"]) for r in recon_t], dtype=float)
            recon_avg = float(np.nanmean(recon_times))
            recon_min = float(np.nanmin(recon_times))
            recon_max = float(np.nanmax(recon_times))
            recon_n = int(np.sum(np.isfinite(recon_times)))
        else:
            recon_avg = recon_min = recon_max = np.nan
            recon_n = 0

        total_runs = len([r for r in run_rows if float(r["target_sim_range_s"]) == float(target)])
        successful_runs = len(rows_t)

        overall_rows.append({
            "target_sim_range_s": target,
            "successful_runs": successful_runs,
            "total_runs": total_runs,
            "achieved_range_avg_s": range_avg,
            "achieved_range_min_s": range_min,
            "achieved_range_max_s": range_max,
            "exploration_total_avg_s": total_avg,
            "exploration_total_min_s": total_min,
            "exploration_total_max_s": total_max,
            "accepted_step_avg_s": step_avg,
            "accepted_step_min_s": step_min,
            "accepted_step_max_s": step_max,
            "steps_completed_avg": steps_avg,
            "single_reconstruction_avg_s": recon_avg,
            "single_reconstruction_min_s": recon_min,
            "single_reconstruction_max_s": recon_max,
            "reconstruction_samples": recon_n,
        })

    write_dict_csv(overall_path, overall_rows)

    group_rows = []
    group_keys = sorted({
        (r["target_sim_range_s"], r["rw_sigma"], r["num_cand"])
        for r in run_rows
    })

    for target, sigma, cand in group_keys:
        rows_g = [
            r for r in successful_rows
            if float(r["target_sim_range_s"]) == float(target)
            and float(r["rw_sigma"]) == float(sigma)
            and int(r["num_cand"]) == int(cand)
        ]

        recon_g = [
            r for r in recon_records
            if float(r["target_sim_range_s"]) == float(target)
            and float(r["rw_sigma"]) == float(sigma)
            and int(r["num_cand"]) == int(cand)
        ]

        if not rows_g:
            continue

        total_times = np.asarray([safe_float(r["exploration_real_elapsed_s"]) for r in rows_g], dtype=float)
        step_times = np.asarray([safe_float(r["accepted_step_runtime_s"]) for r in rows_g], dtype=float)
        achieved_ranges = np.asarray([safe_float(r["achieved_simulation_range_s"]) for r in rows_g], dtype=float)

        if recon_g:
            recon_times = np.asarray([safe_float(r["elapsed_s"]) for r in recon_g], dtype=float)
            recon_avg = float(np.nanmean(recon_times))
            recon_n = int(np.sum(np.isfinite(recon_times)))
        else:
            recon_avg = np.nan
            recon_n = 0

        group_rows.append({
            "target_sim_range_s": target,
            "rw_sigma": sigma,
            "num_cand": cand,
            "successful_runs": len(rows_g),
            "achieved_range_avg_s": float(np.nanmean(achieved_ranges)),
            "exploration_total_avg_s": float(np.nanmean(total_times)),
            "exploration_total_min_s": float(np.nanmin(total_times)),
            "exploration_total_max_s": float(np.nanmax(total_times)),
            "accepted_step_avg_s": float(np.nanmean(step_times)),
            "single_reconstruction_avg_s": recon_avg,
            "reconstruction_samples": recon_n,
        })

    write_dict_csv(group_path, group_rows)

    print_direct_summary(overall_rows, group_rows)

    print("\nSaved:")
    print(run_path)
    print(timing_path)
    print(overall_path)
    print(group_path)


def print_direct_summary(overall_rows, group_rows):
    print("\n" + "=" * 100)
    print("直接统计结果：按目标模拟时间范围汇总，只统计成功达到目标范围的 run")
    print("=" * 100)

    for row in overall_rows:
        target = row["target_sim_range_s"]

        print(f"\n模拟范围 0--{target:g} s")
        print("-" * 80)
        print(f"成功样本数 / 总样本数: {row['successful_runs']} / {row['total_runs']}")
        print(
            f"实际达到模拟范围: "
            f"avg={row['achieved_range_avg_s']:.6f} s, "
            f"min={row['achieved_range_min_s']:.6f} s, "
            f"max={row['achieved_range_max_s']:.6f} s"
        )
        print(
            f"平均整体时间开销: "
            f"avg={row['exploration_total_avg_s']:.6f} s, "
            f"min={row['exploration_total_min_s']:.6f} s, "
            f"max={row['exploration_total_max_s']:.6f} s"
        )
        print(
            f"平均探索单步开销: "
            f"avg={row['accepted_step_avg_s']:.6f} s/step, "
            f"min={row['accepted_step_min_s']:.6f} s/step, "
            f"max={row['accepted_step_max_s']:.6f} s/step"
        )
        print(f"平均完成步数: {row['steps_completed_avg']:.3f}")
        print(
            f"单次重建开销: "
            f"avg={row['single_reconstruction_avg_s']:.6f} s, "
            f"min={row['single_reconstruction_min_s']:.6f} s, "
            f"max={row['single_reconstruction_max_s']:.6f} s"
        )
        print(f"单次重建统计样本数: {row['reconstruction_samples']}")

    print("\n" + "=" * 100)
    print("按参数组统计")
    print("=" * 100)

    for row in group_rows:
        print(
            f"target=0--{row['target_sim_range_s']:g}s | "
            f"sigma={row['rw_sigma']} | cand={row['num_cand']} | "
            f"runs={row['successful_runs']} | "
            f"range_avg={row['achieved_range_avg_s']:.4f}s | "
            f"total_avg={row['exploration_total_avg_s']:.2f}s | "
            f"step_avg={row['accepted_step_avg_s']:.2f}s/step | "
            f"recon_avg={row['single_reconstruction_avg_s']:.5f}s | "
            f"recon_n={row['reconstruction_samples']}"
        )


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-sim-ranges",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0],
        help="Target simulated time ranges. Default: 1 3 5",
    )

    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="Exploration strength values.",
    )

    parser.add_argument(
        "--cand-nums",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Number of candidates per step.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Repeat runs per configuration.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Safety upper bound.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes. Default: 4",
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Disable figure saving to reduce overhead during timing benchmark.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = []
    for target in args.target_sim_ranges:
        for sigma in args.sigmas:
            for cand in args.cand_nums:
                for run_id in range(args.repeat):
                    tasks.append((
                        target,
                        sigma,
                        cand,
                        run_id,
                        args.max_steps,
                        not args.no_figures,
                    ))

    print(f"Total tasks: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print(f"FORCE_CPU: {FORCE_CPU}")
    print(f"Output directory: {OUT_DIR}")

    start_t = time.perf_counter()

    # Windows / Anaconda 下必须放在 if __name__ == "__main__" 保护内调用
    ctx = mp.get_context("spawn")

    run_rows = []
    all_timing_records = []

    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(worker_run, tasks):
            run_rows.append(result["summary"])
            all_timing_records.extend(result["timing_records"])

            # 每完成一个任务就保存一次，防止中途断掉丢数据
            summarize_results(run_rows, all_timing_records, OUT_DIR)

    wall_elapsed = time.perf_counter() - start_t

    summarize_results(run_rows, all_timing_records, OUT_DIR)

    print("\n" + "=" * 100)
    print(f"4-worker benchmark finished. Total wall-clock time: {wall_elapsed:.2f} s")
    print("=" * 100)


if __name__ == "__main__":
    main()