#!/usr/bin/env bash
# Night training queue for microduck_rl on the GPU machine (WSL2 Ubuntu).
#
#   bash training/night_queue.sh training/jobs.txt
#
# Each line of the jobs file is one training run: TASK_ID [extra train options].
# Runs execute one after another, never in parallel (the RTX 4060 has 8 GB).
# For every run:
#   1. smoke test (64 envs, 5 iterations): a broken config costs minutes, not the night
#   2. the real run (tensorboard logger, no wandb)
#   3. export every EXPORT_EVERY-th checkpoint and the last one to ONNX
#   4. pack the run's logs (without the .pt files) for the USB stick
# A failure is written to summary.txt and the queue moves on to the next line.
#
# Ctrl+C once: stop the current run, export what it saved, end the queue.
# Ctrl+C twice: abort immediately.
#
# Settings (environment variables):
#   MICRODUCK_RL   repository                     (default ~/microduck_rl)
#   EXPORT_EVERY   export interval in iterations  (default 1000)
#   SKIP_SMOKE=1   skip the smoke tests
#   NIGHT_OUT      output folder                  (default ~/night_runs/night_<date>_<time>)
#   COPY_TO        copied here at the end         (default "" = no copy; e.g. /mnt/c/runs_out on WSL)

set -u

JOBS=${1:?"usage: bash night_queue.sh JOBS_FILE"}
[ -f "$JOBS" ] || { echo "NG: jobs file not found: $JOBS"; exit 1; }
JOBS=$(realpath "$JOBS")
REPO=${MICRODUCK_RL:-$HOME/microduck_rl}
EXPORT_EVERY=${EXPORT_EVERY:-1000}
OUT=${NIGHT_OUT:-$HOME/night_runs/night_$(date +%Y%m%d_%H%M)}
COPY_TO=${COPY_TO-}
SUMMARY=$OUT/summary.txt

cd "$REPO" 2>/dev/null || { echo "NG: repository not found: $REPO"; exit 1; }
grep -q "mjlab-microduck" pyproject.toml 2>/dev/null || { echo "NG: $REPO is not microduck_rl"; exit 1; }
command -v uv >/dev/null || { echo "NG: uv is not on PATH (run: source ~/.bashrc)"; exit 1; }
mkdir -p "$OUT/onnx"

log() { echo "[$(date '+%m/%d %H:%M:%S')] $*" | tee -a "$SUMMARY"; }
gpu() {
    nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}
# Mjlab-HeadStand-Flat-MicroDuck -> HeadStand
short_name() { echo "$1" | sed -E 's/^Mjlab-//; s/-MicroDuck.*$//; s/-Flat$//'; }

STOP=0
on_int() {
    STOP=$((STOP + 1))
    if [ "$STOP" -ge 2 ]; then
        log "aborted by Ctrl+C"
        exit 130
    fi
    log "Ctrl+C: stopping this run, exporting what it saved, then the queue ends (Ctrl+C again = abort now)"
}
trap on_int INT

log "queue start: $JOBS"
log "repo $REPO @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "gpu: $(gpu)"
log "disk free: $(df -h "$HOME" | awk 'NR == 2 {print $4}')"
log "output: $OUT"

n=0
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}          # tolerate Windows line endings
    line=${line%%#*}            # strip comments
    read -r -a args <<< "$line"
    [ ${#args[@]} -eq 0 ] && continue
    n=$((n + 1))
    task=${args[0]}
    name=$(short_name "$task")
    tag="job${n}_${name}"

    # 1. smoke test (task id only: the real run's options may set num-envs)
    if [ "${SKIP_SMOKE:-0}" != "1" ]; then
        log "$tag smoke test (64 envs, 5 iterations) ..."
        if ! uv run train "$task" --env.scene.num-envs 64 --agent.max-iterations 5 \
                --agent.logger tensorboard --agent.run-name smoke \
                < /dev/null > "$OUT/${tag}_smoke.log" 2>&1; then
            log "$tag SMOKE TEST FAILED -> skipped. Last lines of ${tag}_smoke.log:"
            tail -n 8 "$OUT/${tag}_smoke.log" | sed 's/^/        /' | tee -a "$SUMMARY"
            [ "$STOP" -ge 1 ] && break
            continue
        fi
        log "$tag smoke test OK"
        [ "$STOP" -ge 1 ] && break
        sleep 20
    fi

    # 2. the real run
    marker="$OUT/.${tag}_start"
    touch "$marker"
    sleep 1                     # run folders are named to the second
    log "$tag START: ${args[*]}"
    log "        progress: tail -f $OUT/${tag}.log"
    t0=$(date +%s)
    uv run train "${args[@]}" --agent.logger tensorboard < /dev/null > "$OUT/${tag}.log" 2>&1
    rc=$?
    secs=$(( $(date +%s) - t0 ))
    first=$(grep -ao 'Learning iteration [0-9]*/[0-9]*' "$OUT/${tag}.log" | head -n 1)
    last=$(grep -ao 'Learning iteration [0-9]*/[0-9]*' "$OUT/${tag}.log" | tail -n 1)
    it0=$(echo "$first" | sed -nE 's#.* ([0-9]+)/[0-9]+#\1#p')   # > 0 when resumed
    iters=$(echo "$last" | sed -nE 's#.* ([0-9]+)/[0-9]+#\1#p')
    spi="?"
    [ -n "$iters" ] && spi=$(awk -v s="$secs" -v a="$it0" -v b="$iters" 'BEGIN { printf "%.2f", s / (b - a + 1) }')
    log "$tag END: exit=$rc  time=$((secs / 3600))h$(( (secs % 3600) / 60 ))m  ${last:-no iteration seen}  ~${spi} s/iter"
    log "        gpu after run: $(gpu)"
    if [ "$rc" -ne 0 ] && [ "$STOP" -eq 0 ]; then
        log "        run failed. Last lines of ${tag}.log:"
        tail -n 8 "$OUT/${tag}.log" | sed 's/^/        /' | tee -a "$SUMMARY"
    fi

    # 3 + 4. export checkpoints, pack logs
    mapfile -t runs < <(find logs/rsl_rl -mindepth 2 -maxdepth 2 -type d -newer "$marker" \
                            ! -name '*_smoke' 2>/dev/null | sort)
    [ ${#runs[@]} -eq 0 ] && log "$tag no run folder found under logs/rsl_rl"
    for rd in "${runs[@]}"; do
        run=$(basename "$rd")
        echo "$REPO/$rd" > "$OUT/${tag}_rundir.txt"
        mapfile -t its < <(ls "$rd" | sed -nE 's/^model_([0-9]+)\.pt$/\1/p' | sort -n)
        if [ ${#its[@]} -eq 0 ]; then
            log "$tag no checkpoints in $rd"
        else
            lastit=${its[-1]}
            pick=()
            for it in "${its[@]}"; do
                if [ "$it" -gt 0 ] && [ $((it % EXPORT_EVERY)) -eq 0 ]; then pick+=("$it"); fi
            done
            [[ " ${pick[*]:-} " == *" $lastit "* ]] || pick+=("$lastit")
            ok=0
            for it in "${pick[@]}"; do
                if uv run scripts/export.py "$task" --checkpoint-file "$rd/model_${it}.pt" \
                        --onnx-file "$OUT/onnx/${tag}_${it}.onnx" \
                        < /dev/null >> "$OUT/${tag}_export.log" 2>&1; then
                    ok=$((ok + 1))
                else
                    log "$tag export FAILED: model_${it}.pt (see ${tag}_export.log)"
                fi
                [ "$STOP" -ge 2 ] && break
            done
            log "$tag exported $ok/${#pick[@]} checkpoints to onnx/ (${pick[*]})"
        fi
        tar czf "$OUT/${tag}_logs.tgz" --exclude='*.pt' -C "$(dirname "$rd")" "$run" \
            && log "$tag logs packed: ${tag}_logs.tgz"
    done

    [ "$STOP" -ge 1 ] && break
    sleep 20                    # let the GPU cool down between runs
done < "$JOBS"

log "queue finished"
if [ -n "$COPY_TO" ] && [ -d "$(dirname "$COPY_TO")" ]; then
    mkdir -p "$COPY_TO" && cp -r "$OUT" "$COPY_TO/" \
        && log "copied to $COPY_TO/$(basename "$OUT")"
fi
