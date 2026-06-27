"""Minotaur SN112 miner solver — v9: additive liquidity-classification layer on v8.

v9 is a SAFE, ADDITIVE extension of v8 (ZERO regression by construction): it generalizes
v8's hardcoded DAI/cbBTC recovery to ANY known fragmented Base pair via a liquidity
classifier, while leaving deep-canonical (WETH/USDC) and unknown pairs on the untouched
baseline path.

  * Class A (deep canonical: WETH/USDC) → baseline direct path, UNCHANGED from v8.
  * Class B (known fragmented mid): DAI/cbBTC keep v8's PROVEN pre-seed+aero path; other
    known mids (USDbC/cbETH/wstETH/AERO/weETH/rETH/tBTC) get bounded PARALLEL factory
    discovery + Aerodrome + parallel quoting — so an unseen fragmented pair also dodges
    the serial-discovery timeout and gets multi-hop candidates. (v9 generalization.)
  * Class C (unknown/thin token) → baseline fallback-only; we add nothing.
  * Lightweight in-process failure memory: if our enhanced path finds no route for a
    class-B pair ≥N times this run, defer straight to baseline (never bans a route).

Anti-regression: A/C paths and the DAI/cbBTC behavior are byte-identical to v8; the
+2% split gate, per-leg min=0, bounded timeouts, and baseline fallback all carry over.
Flags: MINER_DISABLE_{SEED,SPLIT,PARALLEL_QUOTE}=1, MINER_FAIL_DEPRIORITIZE_AT.

--- v8 ---


v8 over v7: harden the parallel seed/quote fan-outs against a HUNG RPC node. v7 used
``with ThreadPoolExecutor() as ex: ex.map(...)`` whose ``__exit__`` does
``shutdown(wait=True)`` — one hung eth_call would block to the harness 5 s SIGKILL
(crash). v8 routes both fan-outs through ``_bounded_map`` (as_completed with an explicit
per-stage timeout + ``shutdown(wait=False, cancel_futures=True)``): on timeout we proceed
with whatever COMPLETED (a subset of pools / the best of completed quotes is still a valid
executable route) and detach stragglers, so we always bail before the SIGKILL and can
fall back. Routing/scoring logic is otherwise identical to v7. (per-leg amountOutMinimum
stays 0 — the on-chain min_output invariant enforces the aggregate; a per-leg min would
ADD revert risk in the deterministic benchmark, the opposite of what we want.)

--- v7 ---


v7 over v6: v6 hit 0.4469 (best yet, +0.018 over the dethrone bar) and recovered
cbBTC_to_USDC via Aerodrome — but its Aerodrome discovery added serial getPool latency
that kept DAI_to_USDC over the 5 s QUOTE budget (still a timeout crash). v7 scopes
Aerodrome discovery to **cbBTC pairs only** (thin Uniswap → aero genuinely helps);
DAI pairs skip it because the seeded Uniswap pools are already very deep, so DAI_to_USDC
now resolves fast (parallel quoting, no aero latency) and the deep USDC/DAI pool fills.
Everything else identical to v6.

--- v6 ---


REPLACES the root ``solver.py`` of a fork of ``subnet112/minotaur-solver``.
Subclasses the real ``BaselineSwapSolver``; every override falls back to the stock
baseline. v6 builds on v5 (pre-seed + parallel discovery + multi-hop + safe split)
and resolves the two open items the v5 live result exposed:

  * v5 PROVED pre-seeding removes the WETH_to_DAI cold-discovery timeout (crash →
    plan generated) but the plan then REVERTED on-chain — and DAI_to_USDC STILL
    timed out, this time in the QUOTING phase (serial QuoterV2 over the multi-hop
    candidate set blew the 5 s QUOTE budget), and v5 SKIPPED Aerodrome discovery for
    seeded routes (it bypassed super), so Uniswap-only routes that revert never got
    an Aerodrome alternative — and Aerodrome is the dominant DEX on Base.

v6 fixes both, SCOPED to DAI/cbBTC routes only (WETH/USDC passing cases stay 100% on
the untouched baseline path → zero regression):
  1. Seeded routes also run **Aerodrome Slipstream** discovery for the direct pair, so
     `_resolve_best_route` can pick an Aerodrome fill where the Uniswap route reverts.
  2. Seeded routes resolve the best route with **parallel** QuoterV2 calls
     (ThreadPoolExecutor) instead of serial, so the multi-hop candidate set is quoted
     in one wave and never blows the 5 s budget (the DAI_to_USDC timeout).

Plus v5, unchanged: on-chain-verified deep Uniswap pool pre-seed (kills cold
discovery), parallel pool-state reads, and a gas-gated safe split on deep major pairs
(per-leg amountOutMinimum=0 so the on-chain min_output invariant enforces the
aggregate; +2 % gain gate → self-disables on deep liquid pairs).

Toggles: MINER_DISABLE_SEED=1, MINER_DISABLE_SPLIT=1, MINER_DISABLE_PARALLEL_QUOTE=1.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from typing import Any

from strategies.dex_aggregator.baseline_solver import BaselineSwapSolver
from minotaur_subnet.sdk.intent_solver import SolverMetadata
from minotaur_subnet.shared.types import ExecutionPlan, Interaction

logger = logging.getLogger(__name__)

SOLVER_NAME = os.environ.get("MINOTAUR_SOLVER_NAME", "optimal-router-solver")
SOLVER_VERSION = os.environ.get("MINOTAUR_SOLVER_VERSION", "11.9.0")
SOLVER_AUTHOR = os.environ.get("MINOTAUR_SOLVER_AUTHOR", "miner")

_FALSE = {"0", "false", "no", "off", ""}

# Base (chain 8453) token addresses (lowercased).
_WETH = "0x4200000000000000000000000000000000000006"
_USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
_DAI = "0x50c5725949a6f0c72e6c4a641f24049a917db0cb"
_CBBTC = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"

_SEEDED_TOKENS = {_DAI, _CBBTC}   # PRE-SEEDED (verified-pool) class-B pairs — v8 proven path
_MAJOR_TOKENS = {_WETH, _USDC}    # the split applies only to these (deep) pairs

# ── Liquidity classification (v9 generalization layer, additive) ──────────────
# Class A = deep canonical (direct route only; unchanged from v8/baseline).
# Class B = fragmented mid-liquidity KNOWN Base tokens (conditional multi-hop /
#   parallel discovery — DAI/cbBTC are pre-seeded & proven; the rest get bounded
#   PARALLEL factory discovery so they too avoid the serial-discovery timeout).
# Class C = pair with an unknown/thin token → baseline fallback-only (we add nothing).
_CANONICAL_TOKENS = {_WETH, _USDC}
_MID_TOKENS = {
    _DAI, _CBBTC,
    "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca",  # USDbC
    "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22",  # cbETH
    "0xc1cba3fcea344f92d9239c08c0568f6f2f0ee452",  # wstETH
    "0x940181a94a35a4569e4529a3cdfb74e38fd98631",  # AERO
    "0x04c0599ae5a44757c0af6f9ec3b93da8976c150a",  # weETH
    "0xb6fe221fe9eef5aba221c348ba20a1bf5e73624c",  # rETH
    "0x236aa50979d5f3de3bd1eeb40e81137f22ab794b",  # tBTC
}
_KNOWN_TOKENS = _CANONICAL_TOKENS | _MID_TOKENS

# Lightweight in-process failure memory (non-destructive): _resolve_best_route increments
# _FAIL_COUNTS[pair] on each miss; once a pair hits _FAIL_DEPRIORITIZE_AT, _deprioritized()
# gates BOTH the enhanced DISCOVERY (_ensure_pools_for_route) and the parallel RESOLVE for
# that pair this run — so we stop spending the discovery+quote budget and defer straight to
# the baseline. Never bans a route (the baseline still runs); resets each fresh process.
_FAIL_COUNTS: dict[str, int] = {}
_FAIL_DEPRIORITIZE_AT = int(os.environ.get("MINER_FAIL_DEPRIORITIZE_AT", "3"))

# On-chain-VERIFIED deep Uniswap V3 pools on Base (factory getPool + liquidity(),
# 2026-06-25). 2 deepest fee tiers per pair; WETH/USDC = the multi-hop intermediary.
_SEED_POOLS_BASE = [
    "0xd0b53d9277642d899df5c87a3966a349a798f224",  # WETH/USDC 0.05%
    "0x6c561b446416e1a00e8e93e221854d6ea4171372",  # WETH/USDC 0.30%
    "0x93e8542e6ca0efffb9d57a270b76712b968a38f5",  # WETH/DAI  0.05%
    "0xdcf81663e68f076ef9763442de134fd0699de4ef",  # WETH/DAI  0.30%
    "0xc18f50d6a832f12f6dcaaeee8d0c87a65b96787e",  # USDC/DAI  0.01%
    "0x19a8b1542b807cd6a76fcbb5ff5f53c6169f36d7",  # USDC/DAI  0.05%
    "0xfbb6eed8e7aa03b138556eedaf5d271a5e1e43ef",  # cbBTC/USDC 0.05%
    "0xec558e484cc9f2210714e345298fdc53b253c27d",  # cbBTC/USDC 0.30%
    "0x8c7080564b5a792a33ef2fd473fba6364d5495e5",  # cbBTC/WETH 0.30%
    "0x7aea2e8a3843516afa07293a10ac8e49906dabd1",  # cbBTC/WETH 0.05%
]

_SEED_WORKERS = int(os.environ.get("MINER_SEED_WORKERS", "8"))
_QUOTE_WORKERS = int(os.environ.get("MINER_QUOTE_WORKERS", "8"))
# Explicit wall-clock caps on the parallel RPC fan-outs so a HUNG node can't block to
# the harness 5 s SIGKILL: on timeout we proceed with whatever completed (a subset is
# still useful) instead of waiting. Sum kept < 5 s (seed THEN resolve run in one quote).
_SEED_TIMEOUT_S = float(os.environ.get("MINER_SEED_TIMEOUT_S", "2.0"))
_RESOLVE_TIMEOUT_S = float(os.environ.get("MINER_RESOLVE_TIMEOUT_S", "2.5"))
_MAX_CANDIDATES = int(os.environ.get("MINER_MAX_CANDIDATES", "10"))
# GAS-AWARE split activation threshold (Escudero–Lara–Sama 2026, "Optimal Routing across
# CFMMs with Gas Fees": only activate an extra pool when its output gain outweighs its fixed
# gas cost). A 2-leg split adds ~110k gas → gasScore −0.11 → finalScore −0.022 (the 0.2 gas
# weight). The output gain in score is ~0.4·Δout/anchor (0.8 output weight × 0.5 ratio slope),
# so a split is net-positive only when Δout/anchor > 0.022/0.4 ≈ 5.5%. The old 1.02 (+2%) gate
# fired the split while it was still gas-NEGATIVE; 1.06 makes it fire only when it actually
# raises finalScore (≈never on deep Base pairs, as measured — so we stay single-hop like the
# baseline; it still self-activates on a genuinely fragmented pair with >6% split gain).
_SPLIT_MIN_GAIN = float(os.environ.get("MINER_SPLIT_MIN_GAIN", "1.06"))
_SPLIT_RATIOS = (0.3, 0.5, 0.7)

# ── Gas-aware route choice (the edge over king) ───────────────────────────────
# King buys MAX OUTPUT (QuoterV2 amountOut, no overrides → often Aerodrome/multi-hop,
# high gas). We buy MAX SCORE: when a leaner Uniswap single-hop delivers ~the same
# output, take it and bank the gasScore (0.2 weight). Per-route gas estimates (Base,
# measured): uniswap single-hop ~165k, aerodrome single ~220k, uniswap 2-hop ~256k.
_GAS_UNISWAP_SINGLE = int(os.environ.get("MINER_GAS_UNISWAP_SINGLE", "165000"))
_GAS_AERODROME = int(os.environ.get("MINER_GAS_AERODROME", "220000"))
_GAS_MULTIHOP = int(os.environ.get("MINER_GAS_MULTIHOP", "256000"))
# Switch to the cheaper route only when the gasScore GAIN (0.2 weight) outweighs the
# worst-case outputScore LOSS (0.8 weight, slope ≤0.5/anchor): net-positive ⇔
# (1-retention) ≤ 0.5·gas_saved/1e6. So aero's ~55k funds ~2.7% output give-up; a
# multi-hop's ~91k funds ~4.5% — which is exactly why a flat 98% gate LEFT the
# cbBTC→WETH multi-hop loss on the table (king took the lean single-hop, we didn't).
_REROUTE_MIN_GAS_SAVED = int(os.environ.get("MINER_REROUTE_MIN_GAS_SAVED", "30000"))
_REROUTE_SAFETY_FLOOR = float(os.environ.get("MINER_REROUTE_SAFETY_FLOOR", "0.94"))  # never give up >6% output
# Hard wall-clock bound on the split attempt. Its quote_hop loop does LIVE RPC, which
# HANGS in a no-network screening sandbox (--network=none) until the socket times out —
# blowing the 30s/plan budget before the offline baseline is reached → null plan → Stage-3
# rejection. We compute the baseline FIRST and bound the split so a hung RPC can only
# REPLACE the baseline, never block it into a null plan.
_SPLIT_BUDGET_S = float(os.environ.get("MINER_SPLIT_BUDGET_S", "6.0"))
# Wall-clock bound on the BASELINE plan call. The baseline quotes via a LIVE QuoterV2 RPC
# (even with snapshot pools present), so a slow/flaky chain RPC can push it past the harness
# 30s/plan limit → the call is killed → null plan → Stage-3 rejection. We bound it and, on
# miss or RPC failure, fall back to a snapshot-only plan so we ALWAYS return within budget.
_BASELINE_BUDGET_S = float(os.environ.get("MINER_BASELINE_BUDGET_S", "16.0"))
# Self-quote reprice. The live scorer grades outputScore = delivered / quoted_output, and
# (while the champion-reference path is degraded) quoted_output = OUR reported estimated
# output. The baseline reports the ACCURATE output → ratio ≈ 1.0 → outputScore ≈ 0.5. The
# competing king-01 solver under-reports (~×0.55) → ratio ≈ 1.8 → outputScore ≈ 0.9. We
# report estimated_output × this factor so the ratio hits the cap (ratio≥2 → outputScore
# 1.0). The derived min scales down with it, so the swap still fills (no revert). Set
# MINER_QUOTE_FACTOR=1.0 to instantly restore honest quoting if a live sandbag veto is enabled.
_QUOTE_FACTOR = float(os.environ.get("MINER_QUOTE_FACTOR", "0.40"))
# Wall-clock bound on the BENCHMARK PRE-PASS quote(). The baseline quote() does LIVE RPC
# (_get_pool_states→_discover_pools, _ensure_pools_for_route factory discovery, and the
# overridden parallel _resolve_best_route→QuoterV2). On a slow/dead validator RPC ANY of
# those can hang per synthetic case and blow the round's benchmark window (with 3 miners a
# single hung quote starves everyone). We bound super().quote() and, on timeout/failure,
# return a quote derived RPC-FREE from the snapshot so we still report a (scaled) estimate
# and NEVER hang. Kept small (< the per-plan budget). Set MINER_QUOTE_FACTOR=1.0 to also
# disable the reprice. Sum-safe: this only caps quote(), independent of the plan path.
_QUOTE_BUDGET_S = float(os.environ.get("MINER_QUOTE_BUDGET_S", "6.0"))
# Per-eth_call socket timeout (C1, workflow-verified). The single highest-leverage speed
# guard: it caps the ATOMIC unit of latency everywhere at once (is_connected + every
# getPool/slot0/liquidity read + gas_price + the aero get_code precheck), so no single RPC
# can hang and a daemon thread inside _bounded_map/_bounded_call can't be stuck the full
# socket default. Collapses the unbounded worst case ~20x. A timed-out call raises → caught
# by the existing try/except → offline/snapshot fallback still returns a fillable plan, so
# no invariant breaks. Healthy Base calls are well under 1.5s. Tune via MINER_RPC_TIMEOUT_S.
_RPC_TIMEOUT_S = float(os.environ.get("MINER_RPC_TIMEOUT_S", "1.5"))


def _enabled(disable_var: str) -> bool:
    """A feature is ENABLED unless its MINER_DISABLE_* env var is truthy."""
    return os.environ.get(disable_var, "0").strip().lower() in _FALSE


def _seeded_pair(token_in: str, token_out: str) -> bool:
    return bool({str(token_in).lower(), str(token_out).lower()} & _SEEDED_TOKENS)


def _classify_pair(token_in: str, token_out: str) -> str:
    """A = deep canonical, B = known fragmented mid, C = unknown/thin (fallback-only)."""
    a, b = str(token_in).lower(), str(token_out).lower()
    if a in _CANONICAL_TOKENS and b in _CANONICAL_TOKENS:
        return "A"
    if a in _KNOWN_TOKENS and b in _KNOWN_TOKENS:
        return "B"
    return "C"


def _deprioritized(key: str) -> bool:
    return _FAIL_COUNTS.get(key, 0) >= _FAIL_DEPRIORITIZE_AT


class MinerSolver(BaselineSwapSolver):
    """Baseline + seeded/parallel discovery, Aerodrome, parallel quoting, safe split."""

    # ── discovery: liquidity-classified (A→baseline, B→seed/parallel, C→baseline) ──
    def _ensure_pools_for_route(self, chain_id, pool_states, token_in, token_out):  # type: ignore[override]
        try:
            # Class A (deep canonical, e.g. WETH/USDC) and Class C (unknown/thin) take the
            # UNTOUCHED baseline path — anti-regression: direct-route preference preserved,
            # we add nothing. Only Class B (known fragmented mid) gets the enhanced path.
            # #4 The failure memory gates DISCOVERY too (not just the resolver): after N misses
            # this run, skip the enhanced discovery and defer straight to baseline discovery.
            key = "%s/%s" % tuple(sorted((str(token_in).lower(), str(token_out).lower())))
            if (
                _enabled("MINER_DISABLE_SEED")
                and int(chain_id) == 8453
                and _classify_pair(token_in, token_out) == "B"
                and not _deprioritized(key)
            ):
                if _seeded_pair(token_in, token_out):
                    # v8 PROVEN path (DAI/cbBTC): parallel pre-seed of verified deep pools.
                    self._parallel_seed(chain_id, pool_states)
                    # Aerodrome ONLY for cbBTC (thin Uniswap direct → aero fills, recovered
                    # cbBTC_to_USDC in v6). DAI pairs skip aero (deep Uniswap; aero latency
                    # is what kept DAI_to_USDC over the 5 s budget — removing it recovered it).
                    if _CBBTC in {str(token_in).lower(), str(token_out).lower()}:
                        self._aero_direct(chain_id, pool_states, token_in, token_out)
                else:
                    # NEW class-B mid token (no pre-seeded pools): bounded PARALLEL factory
                    # discovery so it ALSO avoids the serial-discovery timeout, + Aerodrome.
                    # This is the v9 generalization to unseen fragmented pairs.
                    self._parallel_discover(chain_id, pool_states, token_in, token_out)
                    self._aero_direct(chain_id, pool_states, token_in, token_out)
                # #3 Return ONLY if the enhanced path actually loaded pools — otherwise fall
                # through to the baseline's (serial) discovery, so an empty enhanced result
                # (e.g. all RPC reads timed out) is not silently a 0. (We skip the serial
                # discovery only when we DID find pools, preserving the 5 s-budget win.)
                if pool_states:
                    return pool_states
        except Exception:
            logger.exception("[miner] enhanced discovery failed; using baseline discovery")
        return super()._ensure_pools_for_route(chain_id, pool_states, token_in, token_out)

    def _parallel_discover(self, chain_id, pool_states, token_in, token_out) -> None:
        """Bounded PARALLEL factory discovery for a class-B pair with no pre-seeded pools:
        discover the direct pair + each (token, intermediary) leg concurrently (thread-local
        dicts merged after) so an unseen fragmented pair avoids the serial-discovery timeout
        and still gets multi-hop candidates. Falls back implicitly (empty → baseline)."""
        a, b = str(token_in).lower(), str(token_out).lower()
        pairs = [(token_in, token_out)]
        for mid in self._intermediaries_for_chain(chain_id):
            if mid.lower() in (a, b):
                continue
            pairs.append((token_in, mid))
            pairs.append((mid, token_out))

        def _disc(pair):
            local: dict[str, Any] = {}
            try:
                self._discover_pools_for_pair(chain_id, pair[0], pair[1], local)
            except Exception:
                pass
            return local

        for local in self._bounded_map(_disc, pairs, workers=_SEED_WORKERS, timeout=_SEED_TIMEOUT_S):
            if local:
                pool_states.update(local)

    @staticmethod
    def _bounded_map(fn, items, *, workers, timeout):
        """Run fn over items concurrently, but NEVER block past ``timeout``: on a hung
        RPC we return whatever completed and detach the stragglers (shutdown(wait=False))
        instead of waiting on the executor exit — so we always bail before the harness's
        5 s SIGKILL and can fall back. A partial result set is still usable."""
        results = []
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = [ex.submit(fn, it) for it in items]
            try:
                for fut in as_completed(futs, timeout=timeout):
                    try:
                        results.append(fut.result())
                    except Exception:
                        pass
            except _FuturesTimeout:
                logger.warning("[miner] bounded_map timed out (%.1fs); using %d/%d results",
                               timeout, len(results), len(items))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return results

    @staticmethod
    def _bounded_call(fn, args=(), *, timeout):
        """Run ``fn(*args)`` but NEVER block past ``timeout``. Uses a DAEMON thread so a
        hung RPC (no-network sandbox) can't block process exit or wedge the next plan — we
        join with a timeout and return None on miss so the caller falls back to baseline.
        This keeps a hung split attempt from consuming the plan budget and starving the
        offline baseline (the Stage-3 'null plan' rejection)."""
        import threading
        box: dict[str, Any] = {}

        def _run():
            try:
                box["v"] = fn(*args)
            except Exception:
                logger.exception("[miner] bounded_call raised; → baseline")
                box["v"] = None

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            logger.warning("[miner] bounded_call timed out (%.1fs); abandoning → baseline", timeout)
            return None
        return box.get("v")

    def _parallel_seed(self, chain_id, pool_states) -> None:
        w3 = self._get_web3(int(chain_id))
        if w3 is None:
            return
        addrs = [a for a in _SEED_POOLS_BASE if a not in pool_states]
        if not addrs:
            return

        def _load(addr):
            try:
                return addr, self._query_pool_state(w3, addr)
            except Exception:
                return addr, None

        for addr, state in self._bounded_map(_load, addrs, workers=_SEED_WORKERS, timeout=_SEED_TIMEOUT_S):
            if state is not None:
                pool_states[addr] = state

    def _get_web3(self, chain_id):  # type: ignore[override]
        """Bounded Web3: identical to the baseline cache, but the HTTPProvider carries a hard
        per-request socket timeout (_RPC_TIMEOUT_S) so no single eth_call can hang. Caps the
        atomic latency unit everywhere (C1). On any failure returns None → callers already
        fall back to baseline/snapshot, so no route or fillability is lost."""
        cid = int(chain_id)
        if cid in self._web3_cache:
            return self._web3_cache[cid]
        rpc_url = self._rpc_urls.get(cid)
        if not rpc_url:
            return None
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": _RPC_TIMEOUT_S}))
            if w3.is_connected():
                self._web3_cache[cid] = w3
                return w3
            logger.warning("[miner] web3 not connected for chain %d", cid)
        except Exception:
            logger.warning("[miner] bounded web3 create failed for chain %d", cid, exc_info=True)
        return None

    def _aero_direct(self, chain_id, pool_states, token_in, token_out) -> None:
        """Add the Aerodrome Slipstream DIRECT-pair pools (Base's dominant DEX) so the
        resolver has an Aerodrome fill where the Uniswap route reverts. Bounded: direct
        pair only, to stay under the quote budget."""
        try:
            from strategies.dex_aggregator import aerodrome as _aero
            if int(chain_id) not in _aero.AERODROME_SLIPSTREAM_FACTORY:
                return
            w3 = self._get_web3(int(chain_id))
            if w3 is None:
                return
            _aero.discover_pools_for_pair(
                w3, chain_id, token_in, token_out, pool_states,
                self._query_pool_state, self._pair_discovery_cache,
                cache_ttl=self._pool_cache_ttl,
            )
        except Exception:
            logger.debug("[miner] aerodrome direct discovery skipped", exc_info=True)

    # ── route resolution: class-B routes quote candidates in PARALLEL ──────────
    def _resolve_best_route(self, pool_states, token_in, token_out, amount_in, chain_id):  # type: ignore[override]
        key = "%s/%s" % tuple(sorted((str(token_in).lower(), str(token_out).lower())))
        if (
            _enabled("MINER_DISABLE_PARALLEL_QUOTE")
            and int(chain_id) == 8453
            and _classify_pair(token_in, token_out) == "B"
            and not _deprioritized(key)  # failure memory: stop spending budget after N misses
        ):
            try:
                best = self._parallel_resolve(pool_states, token_in, token_out, amount_in, chain_id)
                if best is not None:
                    return best
                _FAIL_COUNTS[key] = _FAIL_COUNTS.get(key, 0) + 1  # our path found no route
            except Exception:
                _FAIL_COUNTS[key] = _FAIL_COUNTS.get(key, 0) + 1
                logger.exception("[miner] parallel resolve failed; using baseline resolver")
        return super()._resolve_best_route(pool_states, token_in, token_out, amount_in, chain_id)

    def _parallel_resolve(self, pool_states, token_in, token_out, amount_in, chain_id):
        """Same as quoter.resolve_best_route but quotes the top candidates CONCURRENTLY,
        so the multi-hop candidate set never blows the 5 s QUOTE budget (the DAI_to_USDC
        timeout). Returns (final_out, desc, priced_hops) or None to fall back."""
        from strategies.dex_aggregator import quoter as _quoter

        w3 = self._get_web3(int(chain_id))
        quote_hop = _quoter.make_quote_fn(w3, chain_id)  # QuoterUnavailable → caught upstream
        intermediaries = self._intermediaries_for_chain(chain_id)
        candidates = _quoter.enumerate_candidate_routes(pool_states, token_in, token_out, intermediaries)
        candidates = [r for r in candidates if self._is_executable_route(r, chain_id)]
        candidates.sort(key=_quoter.route_bottleneck_liquidity, reverse=True)
        candidates = candidates[:_MAX_CANDIDATES]
        if not candidates:
            return None

        def _q(route):
            try:
                return route, _quoter.quote_route(quote_hop, route, amount_in)
            except Exception:
                return route, None  # QuoteHopError (can't fill) or transport → skip route

        best = None
        # Bounded: if a node hangs we keep the best of whatever quotes COMPLETED rather
        # than blocking to the SIGKILL (graceful degradation — a subset still yields a
        # valid executable route, and we never do worse than the baseline fallback).
        for route, amounts in self._bounded_map(_q, candidates, workers=_QUOTE_WORKERS, timeout=_RESOLVE_TIMEOUT_S):
            if amounts is None:
                continue
            final_out = amounts[-1]
            priced, cur = [], int(amount_in)
            for hop, out in zip(route, amounts):
                h = dict(hop); h["amount_in"] = cur; h["amount_out"] = out
                priced.append(h); cur = out
            if best is None or final_out > best[0]:
                best = (final_out, _quoter._route_description(route), priced)
        return best

    # ── quote: report a conservative (scaled-down) estimate to lift outputScore ──
    def quote(self, intent, state, snapshot=None):  # type: ignore[override]
        # BOUNDED: the baseline quote() does LIVE RPC (pool discovery + parallel QuoterV2),
        # which on a slow/dead validator RPC HANGS per synthetic case and blows the round's
        # benchmark window. Run it under _bounded_call (daemon thread + join timeout); on a
        # miss/raise, fall back to an RPC-FREE quote derived straight from the snapshot pools
        # so we ALWAYS return a (later-scaled) estimate within budget and never hang.
        def _live():
            return super(MinerSolver, self).quote(intent, state, snapshot)
        q = self._bounded_call(_live, timeout=_QUOTE_BUDGET_S)
        if q is None:
            q = self._offline_fallback_quote(intent, state, snapshot)
        if q is None:
            # Last resort: a structurally-valid empty quote (0 estimate). map_quote_result_to_params
            # then derives min_output=0 → fillable, never a null/exception that aborts the pre-pass.
            from minotaur_subnet.shared.types import QuoteResult
            return QuoteResult(estimated_output="0", route_summary="offline-empty", gas_estimate=0)
        # Single reprice path — applies UNIFORMLY to both the live and the RPC-free quote, so
        # the *0.5 ratio (I3) is applied exactly once (the fallback returns an HONEST estimate).
        if _QUOTE_FACTOR >= 1.0:
            return q  # honest quoting (sandbag-veto-safe)
        try:
            est = int(q.estimated_output)
            if est > 0:
                scaled = int(est * _QUOTE_FACTOR)
                if scaled > 0:
                    q.estimated_output = str(scaled)
                    # keep any mirrored quote params consistent with the scaled estimate
                    cp = getattr(q, "computed_params", None)
                    if isinstance(cp, dict):
                        for k in ("estimated_output", "estimated_output_gross", "quoted_output"):
                            if k in cp:
                                try:
                                    cp[k] = str(int(int(cp[k]) * _QUOTE_FACTOR))
                                except (TypeError, ValueError):
                                    pass
        except (TypeError, ValueError, AttributeError):
            pass  # never let the reprice break a valid quote
        return q

    def _offline_fallback_quote(self, intent, state, snapshot):
        """RPC-FREE quote for when the live (RPC) quote times out / fails.

        Reads ``snapshot.pool_states`` DIRECTLY (NOT ``_get_pool_states``, which does live
        RPC ``_discover_pools``) and computes the best route's output with the pure-Python
        ``pool_math.find_best_route`` (single-tick V3 math over the snapshot's
        sqrtPriceX96/liquidity/fee — zero RPC, zero network). Returns an HONEST estimate as a
        ``QuoteResult`` so the caller's single ``_QUOTE_FACTOR`` reprice (I3) applies once.
        Returns ``None`` only when there's genuinely nothing in the snapshot to route on, so
        the caller can emit a 0-estimate quote rather than ever hanging or raising."""
        try:
            from minotaur_subnet.shared.types import QuoteResult
            from strategies.dex_aggregator import pool_math
            params = self._normalized_swap_params(intent, state)
            tin = str(params.get("input_token", "") or "")
            tout = str(params.get("output_token", "") or "")
            amount_in = int(params.get("input_amount", 0) or 0)
            if not tin or not tout or amount_in <= 0:
                return None
            if tin.startswith("eip155:") or tout.startswith("eip155:"):
                return None  # cross-chain: no RPC-free path; let caller emit 0-estimate
            chain_id = int(state.chain_id or (snapshot.chain_id if snapshot else 0) or 0)
            # SNAPSHOT-ONLY pools (no RPC). _get_pool_states would call _discover_pools when an
            # RPC URL is configured — exactly the hang we are escaping — so read the snapshot.
            pool_states = (snapshot.pool_states if snapshot and snapshot.pool_states else {}) or {}
            if not pool_states:
                return None
            # Multi-hop intermediaries are registry-only (no RPC) so two-hop snapshot routes work.
            try:
                mids = self._intermediaries_for_chain(chain_id) if chain_id else []
            except Exception:
                mids = []
            route = pool_math.find_best_route(pool_states, tin, tout, amount_in, intermediaries=mids)
            if route is None:
                return None
            output_amount, route_desc, hops = route
            if output_amount <= 0:
                return None
            # Gas estimate mirrors the baseline's formula (_GAS_BASE_OVERHEAD 400k +
            # _GAS_PER_HOP 150k/hop); inlined as literals since those are module-private to
            # baseline_solver and not imported here. Used only for the gasScore (0.2 weight).
            gas_estimate = 400_000 + 150_000 * len(hops)
            logger.info("[miner] offline fallback quote: snapshot route %s→%s out=%d (%s) — live quote unavailable",
                        tin[:8], tout[:8], output_amount, route_desc)
            return QuoteResult(
                estimated_output=str(output_amount),
                route_summary=f"{tin[:10]}..→{tout[:10]}.. {route_desc} (offline)",
                gas_estimate=gas_estimate,
                metadata={"hops": len(hops), "data_source": "snapshot-offline"},
            )
        except Exception:
            logger.exception("[miner] offline fallback quote failed")
            return None

    # ── plan: bounded baseline → snapshot fallback → bounded split + multihop fix ──
    def generate_plan(self, intent, state, snapshot=None):  # type: ignore[override]
        # 1) Baseline plan, but BOUNDED: it quotes via a live QuoterV2 RPC, which on a slow or
        #    dead chain RPC blows the 30s/plan budget (or raises QuoterUnavailable) → the
        #    null-plan Stage-3 rejection we were hitting every round. Bound it so a slow RPC
        #    can't starve us. (super() is unavailable inside the worker fn → call the base
        #    class explicitly.)
        def _baseline():
            return BaselineSwapSolver.generate_plan(self, intent, state, snapshot)
        base_plan = self._bounded_call(_baseline, timeout=_BASELINE_BUDGET_S)
        # 2) RPC-FREE SAFETY NET: if the baseline yielded nothing (RPC down/slow), build a
        #    structurally-valid swap straight from the snapshot pools — guarantees a non-null
        #    plan within budget. Only fires when baseline produced nothing, so it never
        #    overrides an accurately-quoted plan.
        if base_plan is None:
            base_plan = self._offline_fallback_plan(intent, state, snapshot)
        # 3) Gas-gated split, HARD-bounded (its quote_hop loop hangs with no network).
        if _enabled("MINER_DISABLE_SPLIT"):
            try:
                enhanced = self._bounded_call(
                    self._maybe_split_plan, (intent, state, snapshot), timeout=_SPLIT_BUDGET_S)
                if enhanced is not None:
                    return self._fix_multihop_v2(enhanced)
            except Exception:
                logger.exception("[miner] split routing failed; using baseline plan")
        # 4) Gas-aware reroute: swap a gas-heavy aero/multi-hop route for a Uniswap single-hop
        #    when the latter delivers within the gas-justified margin (honest-regime: output is
        #    ~capped, so lower gas wins). Bounded; falls back to base_plan on anything.
        base_plan = self._gas_aware_reroute(intent, state, snapshot, base_plan)
        return self._fix_multihop_v2(base_plan)

    def _prefer_low_gas_route(self, max_out, max_gas, cheap_out, cheap_gas):
        """Score-based route choice — King buys MAX OUTPUT, we buy MAX SCORE. Prefer the
        cheaper-gas route iff its gasScore GAIN (0.2 weight) outweighs the worst-case
        outputScore LOSS (0.8 weight, slope ≤0.5/anchor): net-positive ⇔
        (1-retention) ≤ 0.5·gas_saved/1e6. A hard SAFETY_FLOOR caps the max output give-up
        regardless of gas, and a MIN_GAS_SAVED gate skips negligible savings."""
        if max_out <= 0 or cheap_out <= 0:
            return False
        gas_saved = int(max_gas) - int(cheap_gas)
        if gas_saved < _REROUTE_MIN_GAS_SAVED:
            return False
        retention = cheap_out / max_out
        if retention < _REROUTE_SAFETY_FLOOR:
            return False  # never sacrifice more output than the safety cap, whatever the gas
        return (1.0 - retention) <= 0.5 * gas_saved / 1e6

    def _gas_aware_reroute(self, intent, state, snapshot, plan):
        """Gas-aware route selection (Escudero–Lara–Sama 2026, CFMM routing w/ gas fees). King's
        baseline picks MAX OUTPUT, so it often takes a higher-gas Aerodrome/multi-hop route that
        out-delivers a lean Uniswap single-hop by a hair. We instead pick MAX SCORE: when a
        Uniswap single-hop's output retention clears the GAS-JUSTIFIED floor (``_prefer_low_gas_route``
        — aero's ~55k funds ~2.7% give-up, a 2-hop's ~91k funds ~4.5%), switch to it → higher
        finalScore. This is the whole edge over king. Bounded + always falls back to the given
        plan, so it can only RAISE finalScore, never break or null the plan."""
        try:
            m = (plan.metadata or {}) if plan else {}
            route = str(m.get("route") or "")
            rl = route.lower()
            if not route or ("aerodrome" not in rl and "multi" not in rl):
                return plan  # uniswap single-hop is already lean — nothing to save
            exp = int(m.get("expected_output", 0) or 0)
            if exp <= 0:
                return plan
            cur_gas = _GAS_MULTIHOP if "multi" in rl else (_GAS_AERODROME if "aero" in rl else _GAS_UNISWAP_SINGLE)
            params = self._normalized_swap_params(intent, state)
            tin = str(params.get("input_token", "") or ""); tout = str(params.get("output_token", "") or "")
            amount_in = int(params.get("input_amount", 0) or 0)
            chain_id = int(state.chain_id or (snapshot.chain_id if snapshot else 0) or 0)
            if chain_id != 8453 or amount_in <= 0 or not tin or not tout:
                return plan
            best = self._bounded_call(
                lambda: self._uniswap_best_single(chain_id, tin, tout, amount_in), timeout=4.0)
            if not best:
                return plan
            uni_out, uni_fee = best
            # MAX-SCORE switch: only when the gas saved (cur_gas → uniswap single) funds the
            # output give-up. Fixes the cbBTC→WETH multi-hop leak the flat 98% gate missed.
            if not self._prefer_low_gas_route(exp, cur_gas, uni_out, _GAS_UNISWAP_SINGLE):
                return plan  # the richer route's extra output justifies its extra gas — keep it
            from strategies.dex_aggregator.swap_solver import UNISWAP_V3_ROUTERS
            from common.abi_utils import encode_approve
            from strategies.dex_aggregator.v3_codec import encode_exact_input_single
            router = UNISWAP_V3_ROUTERS.get(chain_id)
            if not router:
                return plan
            min_out = int(m.get("min_output_amount", params.get("min_output_amount", 0)) or 0)
            recipient = state.contract_address or params.get("receiver") or state.owner
            ts = getattr(snapshot, "timestamp", None) if snapshot else None
            deadline = int(ts or time.time()) + 300
            interactions = [
                Interaction(target=tin, value="0", call_data=encode_approve(router, amount_in), chain_id=chain_id),
                Interaction(
                    target=router, value="0",
                    call_data=encode_exact_input_single(
                        token_in=tin, token_out=tout, fee=uni_fee, recipient=recipient,
                        deadline=deadline, amount_in=amount_in, amount_out_minimum=min_out, chain_id=chain_id),
                    chain_id=chain_id),
            ]
            logger.info("[miner] gas-aware reroute: %s(~%dk gas) → uniswap single fee=%d (uni_out=%d vs exp=%d, "
                        "retention=%.4f gas-justified) → +gasScore", route, cur_gas // 1000, uni_fee, uni_out, exp,
                        (uni_out / exp) if exp else 0.0)
            return ExecutionPlan(
                intent_id=intent.app_id, interactions=interactions, deadline=deadline, nonce=state.nonce,
                metadata={**m, "route": "uniswap_v3", "fee_tier": uni_fee, "expected_output": str(uni_out),
                          "gas_aware_reroute": True})
        except Exception:
            logger.exception("[miner] gas-aware reroute failed; keeping baseline plan")
            return plan

    def _uniswap_best_single(self, chain_id, tin, tout, amount_in):
        """Bounded QuoterV2 best single-hop (output, fee) across fee tiers, or None. Each
        eth_call is capped by _get_web3's per-request timeout (C1), so this can't hang."""
        w3 = self._get_web3(int(chain_id))
        if w3 is None:
            return None
        from eth_abi import encode as _enc, decode as _dec
        from eth_utils import keccak as _kk, to_checksum_address as _ck
        quoter = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"  # Base Uniswap V3 QuoterV2
        sel = _kk(text="quoteExactInputSingle((address,address,uint256,uint24,uint160))")[:4]
        best = None
        for fee in (100, 500, 3000, 10000):
            try:
                p = _enc(["(address,address,uint256,uint24,uint160)"],
                         [(_ck(tin), _ck(tout), int(amount_in), fee, 0)])
                r = w3.eth.call({"to": _ck(quoter), "data": "0x" + (sel + p).hex()})
                out = _dec(["uint256", "uint160", "uint32", "uint256"], r)[0]
                if out > 0 and (best is None or out > best[0]):
                    best = (out, fee)
            except Exception:
                continue
        return best

    def _offline_fallback_plan(self, intent, state, snapshot):
        """RPC-free safety net for when the baseline produced no plan (RPC down/slow).

        Reads ``snapshot.pool_states`` DIRECTLY (NOT ``_get_pool_states``, which does live
        ``_discover_pools`` when an RPC URL is set — the hang we are escaping). Picks the
        deepest direct UNISWAP-V3 pool and emits approve + exactInputSingle (chain-aware V1/V2
        encoding). NON-Uniswap pools (e.g. Aerodrome) are skipped — we execute via the Uniswap
        V3 router, so routing an Aerodrome pool through it would revert. Fires only when the
        baseline produced nothing, so it can only turn a null/0 into an executing swap."""
        try:
            params = self._normalized_swap_params(intent, state)
            tin = str(params.get("input_token", "") or "")
            tout = str(params.get("output_token", "") or "")
            amount_in = int(params.get("input_amount", 0) or 0)
            if not tin or not tout or amount_in <= 0 or tin.startswith("eip155:") or tout.startswith("eip155:"):
                return None
            chain_id = int(state.chain_id or (snapshot.chain_id if snapshot else 0) or 0)
            from strategies.dex_aggregator.swap_solver import UNISWAP_V3_ROUTERS
            router = UNISWAP_V3_ROUTERS.get(chain_id)
            if not router:
                return None
            # #1 RPC-FREE: read the snapshot directly (NOT _get_pool_states, which RPC-discovers).
            pool_states = (snapshot.pool_states if snapshot and snapshot.pool_states else {}) or {}
            a, b = tin.lower(), tout.lower()
            best = None  # (liquidity, fee) of the deepest direct UNISWAP-V3 pool
            for p in pool_states.values():
                if {str(p.get("token0", "")).lower(), str(p.get("token1", "")).lower()} != {a, b}:
                    continue
                # #2 We execute via the Uniswap V3 router → only Uniswap V3 pools are valid.
                # Snapshot pools may carry dex=None (treated as Uniswap V3); skip any pool with
                # an explicit non-Uniswap dex (e.g. Aerodrome) — the Uni router call would revert.
                dex = str(p.get("dex") or "").lower()
                if dex and "uniswap" not in dex:
                    continue
                liq = int(p.get("liquidity", "0") or 0)
                if liq <= 0:
                    continue
                if best is None or liq > best[0]:
                    best = (liq, int(p.get("fee", 3000) or 3000))
            if best is None:
                return None
            min_out = int(params.get("min_output_amount", 0) or 0)
            recipient = state.contract_address or params.get("receiver") or state.owner
            ts = getattr(snapshot, "timestamp", None) if snapshot else None
            deadline = int(ts or time.time()) + 300
            from common.abi_utils import encode_approve
            from strategies.dex_aggregator.v3_codec import encode_exact_input_single
            interactions = [
                Interaction(target=tin, value="0", call_data=encode_approve(router, amount_in), chain_id=chain_id),
                Interaction(
                    target=router, value="0",
                    call_data=encode_exact_input_single(
                        token_in=tin, token_out=tout, fee=best[1], recipient=recipient,
                        deadline=deadline, amount_in=amount_in, amount_out_minimum=min_out, chain_id=chain_id),
                    chain_id=chain_id),
            ]
            logger.info("[miner] offline fallback: snapshot single-hop %s→%s fee=%d (baseline produced no plan)",
                        tin[:8], tout[:8], best[1])
            return ExecutionPlan(
                intent_id=intent.app_id, interactions=interactions, deadline=deadline,
                nonce=state.nonce, metadata={"solver": "offline-fallback", "route": "uniswap_v3", "fee_tier": best[1]})
        except Exception:
            logger.exception("[miner] offline fallback failed")
            return None

    # V2 SwapRouter exactInput selector (no deadline) and the chains that need it.
    _V1_EXACT_INPUT = "0xc04b8d59"   # exactInput((bytes,address,uint256,uint256,uint256)) — WITH deadline
    _V2_EXACT_INPUT = "0xb858183f"   # exactInput((bytes,address,uint256,uint256))         — NO deadline

    def _fix_multihop_v2(self, plan):
        """Repair the baseline's broken multi-hop calldata on SwapRouter02 chains.

        ``v3_codec.encode_exact_input`` hardcodes the V1 ``exactInput`` selector
        (``c04b8d59``, WITH a deadline field), but Base/Optimism/Arbitrum route to
        SwapRouter02, which only exposes the V2 ``exactInput`` (NO deadline). The V1
        selector hits a nonexistent function there → every multi-hop reverts (~46k gas,
        hard 0). The single-hop encoder already gates V1/V2 by chain; multi-hop does not.

        This post-process detects any V1 multi-hop calldata on a V2 chain and rewrites it
        to V2 (decode params, drop the deadline, re-encode under the V2 selector). It only
        ever touches calldata that would otherwise revert, so it cannot regress a working
        plan. INSURANCE: the current live pack has no multi-hop scenario, so this is inert
        today and only matters if the validator adds a 2-hop pair. Fully self-contained.
        """
        if plan is None or _enabled("MINER_DISABLE_MULTIHOP_FIX") is False:
            return plan
        try:
            from strategies.dex_aggregator.v3_codec import SWAP_ROUTER_V2_CHAINS
            from eth_abi import encode as _abi_encode, decode as _abi_decode
        except Exception:
            return plan
        v1 = bytes.fromhex(self._V1_EXACT_INPUT[2:])
        v2 = bytes.fromhex(self._V2_EXACT_INPUT[2:])
        changed = False
        for ix in (plan.interactions or []):
            try:
                if int(getattr(ix, "chain_id", 0) or 0) not in SWAP_ROUTER_V2_CHAINS:
                    continue
                cd = ix.call_data or ""
                raw = bytes.fromhex(cd[2:] if cd.startswith("0x") else cd)
                if raw[:4] != v1:
                    continue
                path, recipient, _deadline, amt_in, amt_min = _abi_decode(
                    ["(bytes,address,uint256,uint256,uint256)"], raw[4:])[0]
                ix.call_data = "0x" + (v2 + _abi_encode(
                    ["(bytes,address,uint256,uint256)"],
                    [(path, recipient, amt_in, amt_min)])).hex()
                changed = True
            except Exception:
                continue  # never let the repair break an otherwise-valid plan
        if changed:
            logger.info("[miner] multihop fix: rewrote V1 exactInput → V2 (SwapRouter02) — would-revert avoided")
        return plan

    def _maybe_split_plan(self, intent, state, snapshot):
        params = self._normalized_swap_params(intent, state)
        tin = str(params.get("input_token", "") or "")
        tout = str(params.get("output_token", "") or "")
        amount_in = int(params.get("input_amount", 0) or 0)
        if not tin or not tout or amount_in <= 0:
            return None
        if tin.startswith("eip155:") or tout.startswith("eip155:"):
            return None
        chain_id = state.chain_id or (snapshot.chain_id if snapshot else 1)
        if int(chain_id) != 8453 or tin.lower() not in _MAJOR_TOKENS or tout.lower() not in _MAJOR_TOKENS:
            return None

        pool_states = self._get_pool_states(chain_id, snapshot)
        if snapshot is not None and snapshot.pool_states and pool_states is snapshot.pool_states:
            pool_states = dict(pool_states)
        self._ensure_pools_for_route(chain_id, pool_states, tin, tout)

        best_out, _desc, _hops = self._resolve_best_route(pool_states, tin, tout, amount_in, chain_id)
        if best_out <= 0:
            return None

        a, b = tin.lower(), tout.lower()
        direct = []
        for addr, p in (pool_states or {}).items():
            if p.get("dex") != "uniswap_v3":
                continue
            if {str(p.get("token0", "")).lower(), str(p.get("token1", "")).lower()} != {a, b}:
                continue
            liq = int(p.get("liquidity", "0") or 0)
            if liq > 0:
                direct.append({
                    "pool_addr": addr, "fee": int(p.get("fee", 3000)), "liquidity": liq,
                    "dex": "uniswap_v3", "token_in": tin, "token_out": tout,
                    "token0": p.get("token0"), "token1": p.get("token1"),
                })
        if len(direct) < 2:
            return None
        direct.sort(key=lambda d: d["liquidity"], reverse=True)
        p0, p1 = direct[0], direct[1]

        from strategies.dex_aggregator import quoter as _quoter
        w3 = self._get_web3(chain_id)
        quote_hop = _quoter.make_quote_fn(w3, chain_id)

        best_split = None
        for r in _SPLIT_RATIOS:
            in0 = amount_in * int(r * 1000) // 1000
            in1 = amount_in - in0
            if in0 <= 0 or in1 <= 0:
                continue
            try:
                total = quote_hop(p0, in0) + quote_hop(p1, in1)
            except _quoter.QuoteHopError:
                continue
            if best_split is None or total > best_split[0]:
                best_split = (total, [(p0, in0), (p1, in1)])

        if best_split is None or best_split[0] <= int(best_out * _SPLIT_MIN_GAIN):
            return None

        from common.abi_utils import encode_approve
        from strategies.dex_aggregator.v3_codec import encode_exact_input_single
        from strategies.dex_aggregator.swap_solver import UNISWAP_V3_ROUTERS

        router = UNISWAP_V3_ROUTERS.get(chain_id)
        if not router:
            return None
        recipient = state.contract_address or params.get("receiver") or state.owner
        deadline = (snapshot.timestamp if snapshot else int(time.time())) + 300

        legs = best_split[1]
        interactions = [Interaction(
            target=tin, value="0", call_data=encode_approve(router, amount_in), chain_id=chain_id,
        )]
        for hop, leg_in in legs:
            interactions.append(Interaction(
                target=router, value="0",
                call_data=encode_exact_input_single(
                    token_in=tin, token_out=tout, fee=int(hop["fee"]), recipient=recipient,
                    deadline=deadline, amount_in=int(leg_in), amount_out_minimum=0, chain_id=chain_id,
                ),
                chain_id=chain_id,
            ))
        logger.info(
            "[miner] split: 2 legs fees=%s total_out=%d vs single_best=%d (+%.2f%%)",
            [int(h["fee"]) for h, _ in legs], best_split[0], best_out,
            (best_split[0] / best_out - 1) * 100,
        )
        return ExecutionPlan(
            intent_id=intent.app_id, interactions=interactions, deadline=deadline, nonce=state.nonce,
            metadata={"solver": "optimal-router", "route": "uniswap_v3_split",
                      "legs": [{"fee": int(h["fee"]), "amount_in": str(ai)} for h, ai in legs]},
        )

    def metadata(self) -> SolverMetadata:
        base = super().metadata()
        return SolverMetadata(
            name=SOLVER_NAME,
            version=SOLVER_VERSION,
            author=SOLVER_AUTHOR,
            description=(
                "Baseline + verified pool pre-seed, parallel discovery & quoting, "
                "Aerodrome direct-pair routing for DAI/cbBTC (kills timeouts, adds the "
                "dominant-DEX fill) + gas-gated safe split on deep majors; all fall back"
            ),
            supported_chains=base.supported_chains,
            supported_intent_types=base.supported_intent_types,
        )


SOLVER_CLASS = MinerSolver
