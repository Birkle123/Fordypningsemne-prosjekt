import pyomo.environ as pyo
import numpy as np
import time
from config import config

"""
Experimental stabilized / granular Benders implementation.

Goals:
1. Introduce a trust-region on the linking variable V24 to enforce smaller steps
   between iterations ("smaller step length").
2. Generate multiple cuts per iteration by solving the subproblem at V24_k and
   perturbed points (V24_k ± delta) to build a local bundle of cuts, yielding a
   tighter piecewise linear approximation (greater granularity).
3. (Optional future extension) Could add Magnanti-Wong strengthening by solving
   alternative subproblems at interior points; kept simple here.

Notes:
The second-stage profit as a function of V24 under this hydropower model is
expected to be concave (diminishing marginal value of additional water). For a
maximization problem, Benders cuts of the form:
    alpha <= phi + lambda * (V24 - V24_hat)
are supporting hyperplanes (tangents) to the concave recourse function and form
an outer linear over-estimator. Adding more tangents improves tightness.

We DO NOT make cuts artificially concave by adding quadratic terms; that would
break linearity and classical Benders correctness. Instead, multiple nearby
cuts approximate curvature.
"""


def _build_master(cuts_data, trust_center=None, trust_radius=None):
    m = pyo.ConcreteModel("BendersMasterExperimental")
    m.T = pyo.RangeSet(1, config.T1)

    # First-stage variables
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None))
    m.alpha = pyo.Var(bounds=(-1e6, 1e6))

    # Reservoir balance
    def res_balance(m, t):
        inflow = config.certain_inflow
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * inflow - config.alpha * m.q[t] - m.s[t]
        return m.V[t] == m.V[t-1] + config.alpha * inflow - config.alpha * m.q[t] - m.s[t]
    m.res_balance = pyo.Constraint(m.T, rule=res_balance)

    # Cuts set
    m.Cut = pyo.Set(initialize=cuts_data["Set"])
    m.Phi = pyo.Param(m.Cut, initialize=cuts_data["Phi"])
    m.Lambda = pyo.Param(m.Cut, initialize=cuts_data["Lambda"])  # expected dual
    m.V24_hat = pyo.Param(m.Cut, initialize=cuts_data["V24_hat"])

    def benders_cut(m, c):
        return m.alpha <= m.Phi[c] + m.Lambda[c] * (m.V[config.T1] - m.V24_hat[c])
    m.benders_cuts = pyo.Constraint(m.Cut, rule=benders_cut)

    # Trust region (if provided) to limit movement of V24
    if trust_center is not None and trust_radius is not None:
        m.tr_lower = pyo.Constraint(expr=m.V[config.T1] >= max(0.0, trust_center - trust_radius))
        m.tr_upper = pyo.Constraint(expr=m.V[config.T1] <= min(config.Vmax, trust_center + trust_radius))

    def objective(m):
        first_stage = sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) - config.spillage_cost * sum(m.s[t] for t in m.T)
        return first_stage + m.alpha
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    return m


def _build_subproblem(V24_fixed):
    m = pyo.ConcreteModel("BendersSubproblemExperimental")
    m.T = pyo.RangeSet(config.T1 + 1, config.T)
    m.S = pyo.Set(initialize=config.scenarios)

    m.q = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, None))
    m.V24_hat = pyo.Param(initialize=V24_fixed)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def linking(m, s):
        inflow = config.scenario_info[s]
        return m.V[s, config.T1 + 1] == m.V24_hat + config.alpha * inflow - config.alpha * m.q[s, config.T1 + 1] - m.s[s, config.T1 + 1]
    m.linking = pyo.Constraint(m.S, rule=linking)

    def res(m, s, t):
        if t == config.T1 + 1:
            return pyo.Constraint.Skip
        inflow = config.scenario_info[s]
        return m.V[s, t] == m.V[s, t-1] + config.alpha * inflow - config.alpha * m.q[s, t] - m.s[s, t]
    m.res = pyo.Constraint(m.S, m.T, rule=res)

    def objective(m):
        total = 0
        for s in m.S:
            scen_profit = (
                sum(config.pi[t] * 3.6 * config.E_conv * m.q[s, t] for t in m.T)
                - config.spillage_cost * sum(m.s[s, t] for t in m.T)
                + config.WV_end * m.V[s, config.T]
            )
            total += config.prob[s] * scen_profit
        return total
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    return m


def _solve(model):
    solver = config.get_solver()
    res = solver.solve(model, tee=False)
    return res, model


def _add_cut(cuts_data, sub_model):
    cut_id = len(cuts_data["Set"])
    cuts_data["Set"].append(cut_id)
    cuts_data["Phi"][cut_id] = pyo.value(sub_model.obj)
    dual_sum = 0.0
    for s in config.scenarios:
        dual_sum += config.prob[s] * sub_model.dual[sub_model.linking[s]]
    cuts_data["Lambda"][cut_id] = dual_sum
    cuts_data["V24_hat"][cut_id] = pyo.value(sub_model.V24_hat)
    return cut_id


def run_benders_experimental(iterations=12,
                              perturbation=0.15,
                              trust_radius_initial=0.5,
                              trust_radius_shrink=0.9,
                              min_trust_radius=0.05,
                              add_perturbed_points=True,
                              ring_fractions=(1.0, 0.75, 0.5, 0.25),
                              max_extra_points=12,
                              initial_local_grid=False,
                              local_grid_span=1.0,
                              local_grid_step=0.25,
                              skip_duplicate_tol=1e-4,
                              summary=True):
    """
    Run experimental stabilized Benders.

    Args:
        iterations (int): Max iterations.
        perturbation (float): Delta for extra subproblem points (± around V24_k).
        trust_radius_initial (float): Initial trust region radius for V24.
        trust_radius_shrink (float): Factor to shrink radius each iteration.
        min_trust_radius (float): Minimum radius.
        add_perturbed_points (bool): Whether to solve subproblems at ± perturbation.
        ring_fractions (tuple): Fractions of trust_radius to generate symmetric candidate points.
        max_extra_points (int): Cap on extra candidate solves per iteration.
        initial_local_grid (bool): If True, at first iteration sample a dense local grid around first V24.
        local_grid_span (float): +/- span around center for local grid.
        local_grid_step (float): Spacing for local grid sampling.
        skip_duplicate_tol (float): Tolerance to treat V24 candidates as duplicate existing cuts.
        summary (bool): Print progress.
    Returns:
        dict: Results including sequence of V24, objectives, cuts.
    """
    start = time.time()

    # Data structure for cuts
    cuts_data = {"Set": [], "Phi": {}, "Lambda": {}, "V24_hat": {}}

    trust_center = None
    trust_radius = trust_radius_initial

    seq_V24 = []
    master_objs = []
    true_objs = []
    gaps = []

    existing_v_points = []
    for k in range(iterations):
        m_master = _build_master(cuts_data, trust_center, trust_radius if trust_center is not None else None)
        res_master, m_master = _solve(m_master)
        if res_master.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"Master failed at iter {k+1}: {res_master.solver.termination_condition}")
            break
        V24_k = pyo.value(m_master.V[config.T1])
        alpha_k = pyo.value(m_master.alpha)
        master_obj = pyo.value(m_master.obj)
        seq_V24.append(V24_k)
        master_objs.append(master_obj)

        if summary:
            print(f"ITER {k+1}: Master obj={master_obj:,.2f} V24={V24_k:.3f} alpha={alpha_k:,.2f} cuts={len(cuts_data['Set'])}")

        # Solve subproblem at central point
        sub_central = _build_subproblem(V24_k)
        res_sub_c, sub_central = _solve(sub_central)
        if res_sub_c.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"  Subproblem central infeasible: {res_sub_c.solver.termination_condition}")
            break
        # Central cut (ensure not duplicated)
        central_v = pyo.value(sub_central.V24_hat)
        if not any(abs(central_v - v0) <= skip_duplicate_tol for v0 in existing_v_points):
            _add_cut(cuts_data, sub_central)
            existing_v_points.append(central_v)

        # Build extended candidate set for more granularity
        candidates = []
        if add_perturbed_points:
            # Base +/- perturbation
            for sign in (-1, 1):
                v_test = V24_k + sign * perturbation
                if 0.0 <= v_test <= config.Vmax:
                    candidates.append(v_test)
            # Ring fractions of trust radius (excluding extremes already covered)
            if trust_center is not None and trust_radius > 0:
                for frac in ring_fractions:
                    for sign in (-1, 1):
                        v_ring = trust_center + sign * frac * trust_radius
                        if 0.0 <= v_ring <= config.Vmax:
                            candidates.append(v_ring)
            # Optional local grid first iteration
            if k == 0 and initial_local_grid:
                grid_min = max(0.0, V24_k - local_grid_span)
                grid_max = min(config.Vmax, V24_k + local_grid_span)
                vg = grid_min
                while vg <= grid_max + 1e-9:
                    candidates.append(round(vg, 4))
                    vg += local_grid_step

        # Deduplicate and limit
        filtered = []
        for vpt in candidates:
            if not any(abs(vpt - v0) <= skip_duplicate_tol for v0 in existing_v_points):
                filtered.append(vpt)
        # Sort by distance from center for prioritization
        filtered.sort(key=lambda x: abs(x - V24_k))
        if len(filtered) > max_extra_points:
            filtered = filtered[:max_extra_points]

        # Solve subproblems for filtered candidates
        for v_pt in filtered:
            sub_alt = _build_subproblem(v_pt)
            res_sub_a, sub_alt = _solve(sub_alt)
            if res_sub_a.solver.termination_condition == pyo.TerminationCondition.optimal:
                _add_cut(cuts_data, sub_alt)
                existing_v_points.append(v_pt)
                if summary:
                    print(f"  Added granular cut at V24={v_pt:.3f} phi={pyo.value(sub_alt.obj):,.2f}")

        # Compute true objective using current subproblem (central)
        first_stage_profit = sum(config.pi[t] * 3.6 * config.E_conv * pyo.value(m_master.q[t]) for t in range(1, config.T1 + 1))
        first_stage_profit -= config.spillage_cost * sum(pyo.value(m_master.s[t]) for t in range(1, config.T1 + 1))
        true_obj = first_stage_profit + pyo.value(sub_central.obj)
        true_objs.append(true_obj)
        gap = abs(master_obj - true_obj)
        gaps.append(gap)
        if summary:
            print(f"  True second-stage={pyo.value(sub_central.obj):,.2f} True total={true_obj:,.2f} Gap={gap:,.2f}")

        # Update trust region center and shrink radius (avoid shrinking before enough cuts)
        trust_center = V24_k
        if k > 0:  # skip shrinking at first iteration to allow richer sampling
            trust_radius = max(min_trust_radius, trust_radius * trust_radius_shrink)

        # Simple convergence criterion
        if gap < 1e-6:
            if summary:
                print(f"  Converged (gap < 1e-6) at iteration {k+1}")
            break

    elapsed = time.time() - start

    # Compare with full stochastic model objective (optional)
    try:
        from tasks.task2.stochastic_problem import run_stochastic_problem
        stoch_model = run_stochastic_problem(plot=False, summary=False)
        stoch_obj = pyo.value(stoch_model.obj)
    except Exception:
        stoch_obj = None

    if summary:
        print("=" * 80)
        print("Experimental Benders Results")
        print("=" * 80)
        print(f"Iterations: {len(seq_V24)}")
        print(f"Final master objective: {master_objs[-1]:,.2f} NOK" if master_objs else "No master obj")
        print(f"Final true objective: {true_objs[-1]:,.2f} NOK" if true_objs else "No true obj")
        print(f"Final V24: {seq_V24[-1]:.3f} Mm³" if seq_V24 else "No V24")
        print(f"Final gap: {gaps[-1]:,.6f}" if gaps else "No gap")
        if stoch_obj is not None:
            print(f"Full stochastic objective: {stoch_obj:,.2f} NOK; Difference vs true: {(stoch_obj - true_objs[-1]):,.2f} NOK")
        print(f"Cuts generated: {len(cuts_data['Set'])}")
        print(f"Solve time: {elapsed:.2f} s")
        print("=" * 80)

    return {
        "V24_sequence": seq_V24,
        "master_objectives": master_objs,
        "true_objectives": true_objs,
        "gaps": gaps,
        "cuts": cuts_data,
        "final_V24": seq_V24[-1] if seq_V24 else None,
        "final_true_objective": true_objs[-1] if true_objs else None,
        "stochastic_objective": stoch_obj,
        "solve_time": elapsed,
    }


if __name__ == "__main__":
    run_benders_experimental()
