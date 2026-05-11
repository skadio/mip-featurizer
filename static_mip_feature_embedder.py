"""
Static MIP features from "Algorithm runtime prediction: Methods & evaluation" by Hutter et al. (2014).
Hutter, F., Xu, L., Hoos, H.H. and Leyton-Brown, K., 2014. Algorithm runtime prediction: Methods & evaluation. Artificial Intelligence, 206, pp.79-111.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import gurobipy as gp
import numpy as np
import scipy.sparse as sp


MIPItem = Union[str, gp.Model]
MIPInput = Union[MIPItem, Sequence[MIPItem]]


@dataclass
class StaticMIPFeatures:
    """Container for the ordered 100 static instance-level MIP features."""

    feature_names: List[str]
    feature_values: np.ndarray
    feature_dict: Dict[str, float]


class StaticMIPFeatureEmbedder:
    """Compute 100 static C++-style instance-level features for MIPs."""

    EXPECTED_FEATURE_COUNT = 100

    def __init__(
        self,
        default_for_missing_value: float = -512.0,
        noone: float = 1234.1234,
        eps: float = 1e-10,
    ) -> None:
        self.default_for_missing_value = float(default_for_missing_value)
        self.noone = float(noone)
        self.eps = float(eps)

    def mip_to_static_features(self, input_mips: MIPInput) -> Dict[str, StaticMIPFeatures]:
        """Compute static features for one or more file paths and/or gurobipy models."""
        mip_items = self._get_mip_items(input_mips)
        gurobi_env = self._start_gurobi_env()

        mip_to_features: Dict[str, StaticMIPFeatures] = {}
        for mip_item in mip_items:
            if isinstance(mip_item, gp.Model):
                mip_model = mip_item
                key = getattr(mip_model, "ModelName", "gurobi_model")
                should_dispose = False
            else:
                mip_model = gp.read(mip_item, env=gurobi_env)
                key = mip_item
                should_dispose = True

            try:
                mip_to_features[key] = self._mip_model_to_features(mip_model)
            finally:
                if should_dispose:
                    mip_model.dispose()

        gurobi_env.close()
        return mip_to_features

    def mip_file_to_feature_dict(self, mip_file: str) -> Dict[str, float]:
        """Convenience helper for single-file usage."""
        out = self.mip_to_static_features(mip_file)
        return out[mip_file].feature_dict

    def _mip_model_to_features(self, mip_model: gp.Model) -> StaticMIPFeatures:
        mip_model.update()

        vars_ = mip_model.getVars()
        constrs = mip_model.getConstrs()

        n_vars = int(mip_model.NumVars)
        n_constr = int(mip_model.NumConstrs)

        A = mip_model.getA().tocsr()
        obj = np.array(mip_model.getAttr("Obj", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)
        rhs = np.array(mip_model.getAttr("RHS", constrs), dtype=float) if n_constr > 0 else np.array([], dtype=float)
        senses = np.array(mip_model.getAttr("Sense", constrs), dtype="U1") if n_constr > 0 else np.array([], dtype="U1")
        vtypes = np.array(mip_model.getAttr("VType", vars_), dtype="U1") if n_vars > 0 else np.array([], dtype="U1")
        lbs = np.array(mip_model.getAttr("LB", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)
        ubs = np.array(mip_model.getAttr("UB", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)

        probtype, nq_vars, nq_constr, nq_nzcnt = self._compute_quadratic_metadata(mip_model)

        feature_dict: Dict[str, float] = {
            "probtype": float(probtype),
            "n_vars": float(n_vars),
            "n_constr": float(n_constr),
            "n_nzcnt": float(A.nnz),
            "nq_vars": float(nq_vars),
            "nq_constr": float(nq_constr),
            "nq_nzcnt": float(nq_nzcnt),
        }

        var_type_features, support_size_per_var = self._compute_variable_type_features(
            vtypes=vtypes,
            lbs=lbs,
            ubs=ubs,
            n_vars=n_vars,
        )
        feature_dict.update(var_type_features)

        support_stats = self._basic_vect(
            values=support_size_per_var[support_size_per_var > 0],
            notcount=self.noone,
            feat_name="support_size",
            which_statistics="avg-median-varcoef-q90mq10",
        )
        feature_dict.update(support_stats)

        constraint_set_stats = self._compute_rhs_features_by_sense(rhs=rhs, senses=senses)
        feature_dict.update(constraint_set_stats)

        var_set_indices = [
            np.where(vtypes != "C")[0],
            np.where(vtypes == "C")[0],
            np.arange(n_vars, dtype=int),
        ]

        for s, active_idx in enumerate(var_set_indices):
            graph_out = self._compute_graph_features_for_set(
                A=A,
                rhs=rhs,
                obj=obj,
                active_idx=active_idx,
                var_set_num=s,
                n_constr=n_constr,
            )
            feature_dict.update(graph_out)

        feature_names = self._instance_feature_name_order()
        if len(feature_names) != self.EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Expected {self.EXPECTED_FEATURE_COUNT} features, got {len(feature_names)}."
            )

        feature_values = np.array(
            [feature_dict.get(name, self.default_for_missing_value) for name in feature_names],
            dtype=float,
        )
        ordered_dict = {
            name: float(feature_dict.get(name, self.default_for_missing_value)) for name in feature_names
        }

        return StaticMIPFeatures(
            feature_names=feature_names,
            feature_values=feature_values,
            feature_dict=ordered_dict,
        )

    def _compute_variable_type_features(
        self,
        vtypes: np.ndarray,
        lbs: np.ndarray,
        ubs: np.ndarray,
        n_vars: int,
    ) -> tuple[Dict[str, float], np.ndarray]:
        feature_dict: Dict[str, float] = {}

        num_b = int(np.sum(vtypes == "B"))
        num_i = int(np.sum(vtypes == "I"))
        num_c = int(np.sum(vtypes == "C"))
        num_s = int(np.sum(vtypes == "S"))
        num_n = int(np.sum(vtypes == "N"))

        denom_vars = float(n_vars) if n_vars > 0 else 1.0

        feature_dict.update(
            {
                "num_b_variables": float(num_b),
                "num_i_variables": float(num_i),
                "num_c_variables": float(num_c),
                "num_s_variables": float(num_s),
                "num_n_variables": float(num_n),
                "ratio_b_variables": float(num_b / denom_vars),
                "ratio_i_variables": float(num_i / denom_vars),
                "ratio_c_variables": float(num_c / denom_vars),
                "ratio_s_variables": float(num_s / denom_vars),
                "ratio_n_variables": float(num_n / denom_vars),
            }
        )

        non_cont_mask = vtypes != "C"
        num_i_plus = int(np.sum(non_cont_mask))
        feature_dict["num_i+_variables"] = float(num_i_plus)
        feature_dict["ratio_i+_variables"] = float(num_i_plus / denom_vars)

        is_int_like = np.isin(vtypes, ["I", "N"])
        is_unbounded = (ubs >= 0.5 * gp.GRB.INFINITY) | (lbs <= -0.5 * gp.GRB.INFINITY)
        is_unbounded_disc = is_int_like & is_unbounded

        num_unbounded_disc = int(np.sum(is_unbounded_disc))
        denom_non_cont = float(num_i_plus) if num_i_plus > 0 else 1.0

        feature_dict["num_unbounded_disc"] = float(num_unbounded_disc)
        feature_dict["ratio_unbounded_disc"] = float(num_unbounded_disc / denom_non_cont) if num_i_plus > 0 else 0.0

        support_size_per_var = np.zeros(n_vars, dtype=float)
        for idx in np.where(non_cont_mask)[0]:
            vtype = vtypes[idx]
            if vtype == "B":
                support_size_per_var[idx] = 2.0
            elif vtype in {"I", "N"}:
                if is_unbounded_disc[idx]:
                    continue
                if vtype == "I":
                    support_size_per_var[idx] = (ubs[idx] - lbs[idx] + 1.0)
                else:
                    support_size_per_var[idx] = (ubs[idx] - lbs[idx] + 2.0)
            elif vtype == "S":
                support_size_per_var[idx] = 2.0

        return feature_dict, support_size_per_var

    def _compute_rhs_features_by_sense(self, rhs: np.ndarray, senses: np.ndarray) -> Dict[str, float]:
        feature_dict: Dict[str, float] = {}
        if rhs.size == 0:
            for c in range(3):
                feature_dict[f"rhs_c_{c}_avg"] = self.default_for_missing_value
                feature_dict[f"rhs_c_{c}_varcoef"] = self.default_for_missing_value
            return feature_dict

        sense_map = {0: "<", 1: "=", 2: ">"}
        for c, sense_char in sense_map.items():
            rhs_vals = rhs[senses == sense_char]
            rhs_stats = self._basic_vect(
                values=rhs_vals,
                notcount=self.noone,
                feat_name=f"rhs_c_{c}",
                which_statistics="avg-varcoef",
            )
            feature_dict.update(rhs_stats)
        return feature_dict

    def _compute_graph_features_for_set(
        self,
        A: sp.csr_matrix,
        rhs: np.ndarray,
        obj: np.ndarray,
        active_idx: np.ndarray,
        var_set_num: int,
        n_constr: int,
    ) -> Dict[str, float]:
        t0 = time.perf_counter()

        A_sub = A[:, active_idx] if active_idx.size > 0 else sp.csr_matrix((n_constr, 0), dtype=float)

        vcg_constraint_degree = np.asarray(A_sub.getnnz(axis=1), dtype=float).reshape(-1)
        vcg_constraint_sum = np.asarray(A_sub.sum(axis=1), dtype=float).reshape(-1)

        vcg_var_degree_active = np.asarray(A_sub.getnnz(axis=0), dtype=float).reshape(-1)
        vcg_var_sum_active = np.asarray(A_sub.sum(axis=0), dtype=float).reshape(-1)

        a_normalized_varcoefs = np.zeros(n_constr, dtype=float)
        A_ij_normalized_vals: List[float] = []

        indptr = A_sub.indptr
        data = A_sub.data

        for i in range(n_constr):
            start, end = indptr[i], indptr[i + 1]
            row_vals = data[start:end]
            if row_vals.size == 0:
                a_normalized_varcoefs[i] = 0.0
                continue

            if abs(rhs[i]) > 1e-6:
                A_ij_normalized_vals.extend((row_vals / rhs[i]).tolist())

            abs_vals = np.abs(row_vals)
            denom = float(np.sum(abs_vals))
            if denom <= self.eps:
                a_normalized_varcoefs[i] = 0.0
            else:
                normed = abs_vals / denom
                a_normalized_varcoefs[i] = self._varcoef(normed)

        A_ij_normalized = np.asarray(A_ij_normalized_vals, dtype=float)

        obj_active = np.abs(obj[active_idx]) if active_idx.size > 0 else np.array([], dtype=float)

        obj_per_constr_active = np.zeros_like(obj_active)
        obj_per_sqr_active = np.zeros_like(obj_active)

        constrained_mask = vcg_var_degree_active > 0
        obj_per_constr_active[constrained_mask] = (
            obj_active[constrained_mask] / vcg_var_degree_active[constrained_mask]
        )
        obj_per_sqr_active[constrained_mask] = (
            obj_active[constrained_mask] / np.sqrt(vcg_var_degree_active[constrained_mask])
        )

        s = str(var_set_num)
        out: Dict[str, float] = {}

        out.update(
            self._basic_vect(
                values=vcg_constraint_degree,
                notcount=self.noone,
                feat_name=f"vcg_constr_deg{s}",
                which_statistics="avg-median-varcoef-q90mq10",
            )
        )
        out.update(
            self._basic_vect(
                values=vcg_var_degree_active,
                notcount=self.noone,
                feat_name=f"vcg_var_deg{s}",
                which_statistics="avg-median-varcoef-q90mq10",
            )
        )
        out.update(
            self._basic_vect(
                values=vcg_constraint_sum,
                notcount=self.noone,
                feat_name=f"vcg_constr_weight{s}",
                which_statistics="avg-varcoef",
            )
        )
        out.update(
            self._basic_vect(
                values=vcg_var_sum_active,
                notcount=self.noone,
                feat_name=f"vcg_var_weight{s}",
                which_statistics="avg-varcoef",
            )
        )
        out.update(
            self._basic_vect(
                values=A_ij_normalized,
                notcount=self.noone,
                feat_name=f"A_ij_normalized{s}",
                which_statistics="avg-varcoef",
            )
        )
        out.update(
            self._basic_vect(
                values=a_normalized_varcoefs,
                notcount=self.noone,
                feat_name=f"a_normalized_varcoefs{s}",
                which_statistics="avg-varcoef",
            )
        )
        out.update(
            self._basic_vect(
                values=obj_active,
                notcount=self.noone,
                feat_name=f"obj_coefs{s}",
                which_statistics="avg-std",
            )
        )
        out.update(
            self._basic_vect(
                values=obj_per_constr_active[constrained_mask],
                notcount=self.noone,
                feat_name=f"obj_coef_per_constr{s}",
                which_statistics="avg-std",
            )
        )
        out.update(
            self._basic_vect(
                values=obj_per_sqr_active[constrained_mask],
                notcount=self.noone,
                feat_name=f"obj_coef_per_sqr_constr{s}",
                which_statistics="avg-std",
            )
        )

        out[f"time_VCG{s}"] = float(time.perf_counter() - t0)
        return out

    def _instance_feature_name_order(self) -> List[str]:
        names: List[str] = [
            "probtype",
            "n_vars",
            "n_constr",
            "n_nzcnt",
            "nq_vars",
            "nq_constr",
            "nq_nzcnt",
            "num_b_variables",
            "num_i_variables",
            "num_c_variables",
            "num_s_variables",
            "num_n_variables",
            "ratio_b_variables",
            "ratio_i_variables",
            "ratio_c_variables",
            "ratio_s_variables",
            "ratio_n_variables",
            "num_i+_variables",
            "ratio_i+_variables",
            "num_unbounded_disc",
            "ratio_unbounded_disc",
            "support_size_avg",
            "support_size_median",
            "support_size_varcoef",
            "support_size_q90mq10",
            "rhs_c_0_avg",
            "rhs_c_0_varcoef",
            "rhs_c_1_avg",
            "rhs_c_1_varcoef",
            "rhs_c_2_avg",
            "rhs_c_2_varcoef",
        ]

        for s in range(3):
            names.extend(
                [
                    f"vcg_constr_deg{s}_avg",
                    f"vcg_constr_deg{s}_median",
                    f"vcg_constr_deg{s}_varcoef",
                    f"vcg_constr_deg{s}_q90mq10",
                    f"vcg_var_deg{s}_avg",
                    f"vcg_var_deg{s}_median",
                    f"vcg_var_deg{s}_varcoef",
                    f"vcg_var_deg{s}_q90mq10",
                    f"vcg_constr_weight{s}_avg",
                    f"vcg_constr_weight{s}_varcoef",
                    f"vcg_var_weight{s}_avg",
                    f"vcg_var_weight{s}_varcoef",
                    f"A_ij_normalized{s}_avg",
                    f"A_ij_normalized{s}_varcoef",
                    f"a_normalized_varcoefs{s}_avg",
                    f"a_normalized_varcoefs{s}_varcoef",
                    f"obj_coefs{s}_avg",
                    f"obj_coefs{s}_std",
                    f"obj_coef_per_constr{s}_avg",
                    f"obj_coef_per_constr{s}_std",
                    f"obj_coef_per_sqr_constr{s}_avg",
                    f"obj_coef_per_sqr_constr{s}_std",
                    f"time_VCG{s}",
                ]
            )

        return names

    def _compute_quadratic_metadata(self, mip_model: gp.Model) -> tuple[int, int, int, int]:
        is_mip = bool(int(mip_model.IsMIP))

        q_obj_nnz = 0
        q_obj_var_count = 0

        try:
            q_obj = mip_model.getQ()
            if q_obj is not None:
                q_obj_coo = q_obj.tocoo()
                if q_obj_coo.nnz > 0:
                    nonzero_mask = np.abs(q_obj_coo.data) > self.eps
                    q_obj_nnz = int(np.count_nonzero(nonzero_mask))
                    if q_obj_nnz > 0:
                        touched = np.concatenate([q_obj_coo.row[nonzero_mask], q_obj_coo.col[nonzero_mask]])
                        q_obj_var_count = int(np.unique(touched).size)
        except Exception:
            q_obj_nnz = 0
            q_obj_var_count = 0

        nq_constr = int(mip_model.NumQConstrs)
        q_constr_nnz = 0

        if nq_constr > 0:
            for qc in mip_model.getQConstrs():
                try:
                    q_constr_nnz += self._count_quadratic_terms(mip_model.getQCRow(qc))
                except Exception:
                    continue

        if nq_constr > 0:
            probtype = 5 if is_mip else 4
        elif q_obj_nnz > 0:
            probtype = 3 if is_mip else 2
        else:
            probtype = 1 if is_mip else 0

        nq_vars = q_obj_var_count
        nq_nzcnt = int(q_obj_nnz + q_constr_nnz)
        return probtype, nq_vars, nq_constr, nq_nzcnt

    def _count_quadratic_terms(self, qc_row_obj) -> int:
        if qc_row_obj is None:
            return 0

        stack = [qc_row_obj]
        total = 0

        while stack:
            cur = stack.pop()
            if cur is None:
                continue

            if isinstance(cur, (tuple, list)):
                stack.extend(cur)
                continue

            cls_name = cur.__class__.__name__.lower()
            if "quadexpr" in cls_name:
                size_fn = getattr(cur, "size", None)
                if callable(size_fn):
                    try:
                        total += int(size_fn())
                    except Exception:
                        pass

        return total

    def _basic_vect(
        self,
        values: np.ndarray,
        notcount: float,
        feat_name: str,
        which_statistics: str,
    ) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float).reshape(-1)

        if abs(notcount - self.noone) > 1e-10:
            filtered = arr[np.abs(arr - notcount) > 1e-10]
        else:
            filtered = arr

        mysum = 0.0
        mymean = self.default_for_missing_value
        mystd = self.default_for_missing_value
        mymin = self.default_for_missing_value
        mymax = self.default_for_missing_value
        myq10 = self.default_for_missing_value
        myq25 = self.default_for_missing_value
        mymedian = self.default_for_missing_value
        myq75 = self.default_for_missing_value
        myq90 = self.default_for_missing_value
        myvarcoef = self.default_for_missing_value
        myinvvarcoef = self.default_for_missing_value

        n = int(filtered.size)
        if n > 0:
            mysum = float(np.sum(filtered))
            mymean = float(mysum / n)
            mystd = float(np.sqrt(np.mean((filtered - mymean) ** 2)))

            sorted_vals = np.sort(filtered)
            mymin = float(sorted_vals[0])
            myq10 = float(sorted_vals[n // 10])
            myq25 = float(sorted_vals[n // 4])
            mymedian = float(sorted_vals[n // 2])
            myq75 = float(sorted_vals[(3 * n) // 4])
            myq90 = float(sorted_vals[(9 * n) // 10])
            mymax = float(sorted_vals[-1])

            myvarcoef = self._varcoef(filtered)
            std_for_inv = 1e-10 if abs(mystd) < 1e-10 else mystd
            myinvvarcoef = float(mymean / std_for_inv)

        output: Dict[str, float] = {}
        for token in which_statistics.split("-"):
            if token == "avg":
                output[f"{feat_name}_{token}"] = mymean
            elif token == "sum":
                output[f"{feat_name}_{token}"] = mysum
            elif token == "std":
                output[f"{feat_name}_{token}"] = mystd
            elif token == "min":
                output[f"{feat_name}_{token}"] = mymin
            elif token == "max":
                output[f"{feat_name}_{token}"] = mymax
            elif token == "median":
                output[f"{feat_name}_{token}"] = mymedian
            elif token == "q10":
                output[f"{feat_name}_{token}"] = myq10
            elif token == "q25":
                output[f"{feat_name}_{token}"] = myq25
            elif token == "q75":
                output[f"{feat_name}_{token}"] = myq75
            elif token == "q90":
                output[f"{feat_name}_{token}"] = myq90
            elif token == "varcoef":
                output[f"{feat_name}_{token}"] = myvarcoef
            elif token == "invvarcoef":
                output[f"{feat_name}_{token}"] = myinvvarcoef
            elif token == "q75dq25":
                if self._is_missing(myq75) or self._is_missing(myq25):
                    value = self.default_for_missing_value
                elif myq75 < 1e-6:
                    value = 0.0
                elif myq25 < 1e-6:
                    value = self.default_for_missing_value
                else:
                    value = float(myq75 / myq25)
                output[f"{feat_name}_{token}"] = value
            elif token == "q75mq25":
                if self._is_missing(myq75) or self._is_missing(myq25):
                    value = self.default_for_missing_value
                else:
                    value = float(myq75 - myq25)
                output[f"{feat_name}_{token}"] = value
            elif token == "q90mq10":
                if self._is_missing(myq90) or self._is_missing(myq10):
                    value = self.default_for_missing_value
                else:
                    value = float(myq90 - myq10)
                output[f"{feat_name}_{token}"] = value
            elif token == "maxmmin":
                if self._is_missing(mymax) or self._is_missing(mymin):
                    value = self.default_for_missing_value
                else:
                    value = float(mymax - mymin)
                output[f"{feat_name}_{token}"] = value

        return output

    def _varcoef(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return self.default_for_missing_value

        mean = float(np.mean(values))
        std = float(np.sqrt(np.mean((values - mean) ** 2)))
        mean_for_varcoef = 1e-10 if abs(mean) < 1e-10 else mean
        return float(std / mean_for_varcoef)

    def _is_missing(self, value: float) -> bool:
        return abs(float(value) - self.default_for_missing_value) < 1e-10

    @staticmethod
    def _get_mip_items(input_mips: MIPInput) -> List[MIPItem]:
        inputs = input_mips if isinstance(input_mips, (list, tuple)) else [input_mips]
        mip_items: List[MIPItem] = []
        for item in inputs:
            if isinstance(item, gp.Model):
                mip_items.append(item)
            elif isinstance(item, str) and os.path.isdir(item):
                mip_items.extend(StaticMIPFeatureEmbedder._get_only_mip_files(item))
            elif isinstance(item, str) and os.path.isfile(item):
                mip_items.append(item)
            else:
                raise ValueError(
                    f"Input {item!r} is neither a directory, a file, nor a gurobipy model instance."
                )
        return mip_items

    @staticmethod
    def _get_only_mip_files(input_mip_folder: str, is_sort_by_size: bool = False) -> List[str]:
        all_filenames = os.listdir(input_mip_folder)
        all_filepaths = [os.path.join(input_mip_folder, filename) for filename in all_filenames]
        mip_filepaths = [
            p
            for p in all_filepaths
            if p.lower().endswith(".mps")
            or p.lower().endswith(".lp")
            or p.lower().endswith(".mps.gz")
            or p.lower().endswith(".lp.gz")
        ]
        if is_sort_by_size:
            mip_filepaths = sorted(mip_filepaths, key=os.path.getsize)
        return mip_filepaths

    @staticmethod
    def _start_gurobi_env() -> gp.Env:
        gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam("OutputFlag", 0)
        gurobi_env.start()
        return gurobi_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the standalone 100 static MIP features for LP/MPS files."
    )
    parser.add_argument(
        "input_mips",
        nargs="+",
        help="One or more .lp/.mps files, or directories containing them.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save JSON output.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2).",
    )
    args = parser.parse_args()

    embedder = StaticMIPFeatureEmbedder()
    input_payload: MIPInput = args.input_mips if len(args.input_mips) > 1 else args.input_mips[0]
    mip_to_features = embedder.mip_to_static_features(input_payload)

    payload = {
        key: {
            "feature_count": len(features.feature_names),
            "feature_names": features.feature_names,
            "feature_values": [float(v) for v in features.feature_values.tolist()],
            "feature_dict": features.feature_dict,
        }
        for key, features in mip_to_features.items()
    }

    text = json.dumps(payload, indent=args.indent)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()