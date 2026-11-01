# =============================================================================
#  SDDP model for bundled flood-control storage allocation (pseudocode)
#  Follows the formulation in Sections 2 and 3 of the paper.
#  Symbols: k = reservoir, t = stage (month), g = VRE regime (Markov state)
# =============================================================================

# -----------------------------------------------------------------------------
# Inputs (supplied by the user; see README)
# -----------------------------------------------------------------------------
#   P[t]            regime transition matrices (rows sum to one)
#   Ω[t,g], p[t,g]  stagewise outcomes ω = (ε, R) and probabilities
#                   (inflow innovation × within-regime VRE disturbance)
#   μ[k,t], σ[k,t]  monthly mean and standard deviation of inflow
#   φ               AR(1) coefficient of the standardized inflow anomaly
#   reservoir data  storage bounds, normal-pool storage, flood-control capacity,
#                   discharge and output limits, conversion factors, minimum flow
#   B[b,t]          aggregate flood-control requirement of bundling group b
#   F_fix[k,t]      fixed flood-control requirement (unbundled scheme)
#   P_plan[t]       planned output of the base
#   c_s, c_mf, c_fc, c_end   penalty coefficients

# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
function build_model(scheme)                      # scheme ∈ {:bundled, :unbundled}
    return MarkovianPolicyGraph(transition_matrices = P, sense = :Max) do sp, (t, g)

        # State variables
        @state  w[k]                              # reservoir storage
        @state  z[k]                              # standardized inflow anomaly

        # Decision variables
        @variable  qe[k] ≥ 0, qs[k] ≥ 0          # generation and spillage discharge
        @variable  h[k] ≥ 0                      # hydropower output
        @variable  R_cur ≥ 0, R_def ≥ 0          # VRE curtailment and deficit
        @variable  σ_mf[k] ≥ 0, σ_fc ≥ 0, σ_end[k] ≥ 0   # slack variables

        # Stagewise uncertainty
        parameterize(sp, Ω[t,g], p[t,g]) do (ε, R)
            fix inflow innovation ε[k] and available VRE output R
        end

        # Inflow model
        z[k].out == φ * z[k].in + ε[k]                         # AR(1) transition
        a[k]     =  μ[k,t] + σ[k,t] * z[k].out                 # inverse standardization
        z[k].out ≥ -μ[k,t] / σ[k,t]                            # non-negative inflow

        # Reservoir operation
        In[k]    =  a[k] + Σ_{u ∈ U(k)} (qe[u] + qs[u])        # routing
        w[k].out == w[k].in + (In[k] - qe[k] - qs[k]) * D[t] - Ev[k,t]
        h[k]     == κ[k,t] * qe[k]
        operational bounds on qe, qe + qs, h, w

        # Power balance against the planned output
        Σ_k h[k] + R - R_cur + R_def == P_plan[t]
        R_cur ≤ R

        # Comprehensive water use (soft)
        qe[k] + qs[k] + σ_mf[k] ≥ Q_mf[k,t]

        # Flood-control storage in flood-season stages (soft)
        if t ∈ flood_stages
            f[k] = W_normal[k] - w[k].out
            0 ≤ f[k] ≤ F_max[k]
            if scheme == :bundled
                Σ_{k ∈ K_b} f[k] + σ_fc ≥ B[b,t]                 # group-shared requirement
            else
                f[k] + σ_fc ≥ F_fix[k,t]                         # rigid per-reservoir requirement
            end
        end

        # End-of-horizon storage (soft)
        if t == L
            w[k].out + σ_end[k] ≥ W_end[k]
        end

        # Stage objective, Eq. (20)
        @stageobjective(sp,
            (Σ_k (h[k] - c_s * κ[k,t] * qs[k]) + R - R_cur - R_def) * D[t]
            - c_mf * Σ_k σ_mf[k] - c_fc * σ_fc - c_end * Σ_k σ_end[k])
    end
end

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
#   repeat
#     forward pass : sample regime paths and outcomes, solve the stage
#                    subproblems sequentially, record trial states (ŵ, ẑ)
#     backward pass: re-solve next-stage nodes over all outcomes at (ŵ, ẑ),
#                    average the duals of the storage and AR(1) constraints,
#                    and add a Benders cut to each visited node
#   until the bounds stabilize
for scheme in (:bundled, :unbundled)
    model[scheme] = build_model(scheme)
    train(model[scheme]; stopping_rule)
end

# -----------------------------------------------------------------------------
# Paired out-of-sample evaluation (common random numbers)
# -----------------------------------------------------------------------------
scenarios = sample_paths(P, Ω, p; n = N, seed)        # same paths for both schemes
for scheme in (:bundled, :unbundled)
    results[scheme] = simulate(model[scheme], scenarios)
end
compare(results[:bundled], results[:unbundled])       # paired differences, tests, CVaR
