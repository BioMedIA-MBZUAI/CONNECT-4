import math
import re
import openpyxl
from openpyxl.utils import range_boundaries


class SubcorticalNorms:
    """
    Use the original mmc2.xlsm workbook to compute:
      - predicted (normative) subcortical volumes
      - 95% prediction intervals

    Parameters
    ----------
    workbook_path : str
        Path to mmc2.xlsm.
    """

    def __init__(self, workbook_path: str):
        self.workbook_path = workbook_path

        # One copy with formulas (for structure) and one with numbers only
        self.wb = openpyxl.load_workbook(workbook_path, data_only=False, keep_vba=True)
        self.wb_vals = openpyxl.load_workbook(workbook_path, data_only=True, keep_vba=False)

        self.ws_stats = self.wb["Statistics"]
        self.ws_matrix = self.wb["Matrix"]
        self.ws_matrix_vals = self.wb_vals["Matrix"]

        self.defined = {dn.name: dn for dn in self.wb.defined_names.definedName}

        # Build models (one per region)
        self.region_models = self._build_region_models()

    # ---------- internal helpers ----------

    def _get_range_cells(self, range_str, wb):
        """Return a 2D list of values for a range like 'Matrix!$C$33:$N$44'."""
        sheetname, ref = range_str.split("!")
        sheetname = sheetname.strip("'")
        ws = wb[sheetname]
        min_col, min_row, max_col, max_row = range_boundaries(ref)
        cells = []
        for r in range(min_row, max_row + 1):
            row = []
            for c in range(min_col, max_col + 1):
                row.append(ws.cell(row=r, column=c).value)
            cells.append(row)
        return cells

    def _build_region_models(self):
        """
        Read everything we need from the workbook and return
        a dict: { 'Accumbens L': {...}, ... }.
        """
        models = {}

        for row in range(11, 37):
            label = self.ws_stats.cell(row=row, column=1).value
            if not label:
                continue

            formula_c = self.ws_stats.cell(row=row, column=3).value
            if not isinstance(formula_c, str) or "MMULT" not in formula_c:
                continue

            # Example C-cell formula:
            # =IF(..., "-", MMULT(X_Left_Accumbens_area,B_Left_Accumbens_area))
            # or for ventricles:
            # =IF(..., "-", 10^MMULT(X_ventricles_log,B_ventricles_log))
            m = re.search(r"(B_[A-Za-z0-9_]+)", formula_c)
            if not m:
                continue

            b_name = m.group(1)              # e.g. 'B_Left_Accumbens_area'
            model_key = b_name[2:]           # e.g. 'Left_Accumbens_area'

            pred_name = "pred_" + model_key  # row with term names
            m_name = "M_" + model_key        # covariance matrix

            # Term names (B0, agec, tivc, ...)
            beta_term_cells = self._get_range_cells(self.defined[pred_name].attr_text,
                                                    self.wb)
            beta_names = [v for v in beta_term_cells[0] if v is not None]

            # Coefficients β
            beta_vals = [
                row_vals[0]
                for row_vals in self._get_range_cells(self.defined[b_name].attr_text,
                                                      self.wb_vals)
            ]

            # Covariance matrix M
            M_vals = self._get_range_cells(self.defined[m_name].attr_text,
                                           self.wb_vals)

            # Find the row in Matrix!A: that matches this model_key
            matrix_row_idx = None
            for r in range(1, self.ws_matrix.max_row + 1):
                if self.ws_matrix.cell(row=r, column=1).value == model_key:
                    matrix_row_idx = r
                    break
            if matrix_row_idx is None:
                continue

            n = self.ws_matrix_vals.cell(row=matrix_row_idx, column=3).value
            mse = self.ws_matrix_vals.cell(row=matrix_row_idx, column=4).value
            t_val = self.ws_matrix_vals.cell(row=matrix_row_idx, column=2).value

            # Is this a log10 model? (all ventricles are)
            log10_model = isinstance(formula_c, str) and "10^MMULT" in formula_c

            models[label] = dict(
                label=label,
                model_key=model_key,
                beta_names=beta_names,
                beta_vals=beta_vals,
                M=M_vals,
                n=n,
                mse=mse,
                t=t_val,
                log10_model=log10_model,
            )

        return models

    @staticmethod
    def _build_covariates(age, sex, field_strength, manufacturer, icv):
        """
        Construct covariates on the same scale used in Matrix!E2:U2.

        Parameters
        ----------
        age : float
        sex : int           (Male = 1, Female = 0)
        field_strength : int
            1 for 1.5T, 0 for 3T  (as in the Statistics sheet)
        manufacturer : int
            GE = 1, Philips = 2, Siemens = 3
        icv : float
            Estimated intracranial volume (mm^3).
        """
        # Centering constants from Matrix!F2 and J2
        mean_age = 47.5634617
        mean_tiv = 1521907.28

        agec = age - mean_age
        tivc = icv - mean_tiv
        sexq = sex
        MFSq = field_strength
        GE = 1 if manufacturer == 1 else 0
        Philips = 1 if manufacturer == 2 else 0

        vals = {}

        def add(name, value):
            # store only lower-case keys; we’ll look up by lower-case
            vals[name.lower()] = value

        add("B0", 1.0)
        add("agec", agec)
        add("agec2", agec ** 2)
        add("agec3", agec ** 3)
        add("sexq", sexq)
        add("tivc", tivc)
        add("tivc2", tivc ** 2)
        add("tivc3", tivc ** 3)
        add("MFSq", MFSq)
        add("GE", GE)
        add("Philips", Philips)
        add("GE_X_MFSq", GE * MFSq)
        add("PHILIPS_X_MFSq", Philips * MFSq)
        add("tivc_X_MFSq", tivc * MFSq)
        add("AGEc_X_SEXq", agec * sexq)
        add("tivc_X_GE", tivc * GE)
        add("tivc_X_Philips", tivc * Philips)

        return vals

    # ---------- public API ----------

    def regions(self):
        """Return the list of available region labels (as shown in the Excel sheet)."""
        return sorted(self.region_models.keys())

    def predict_region(self, region_label, age, sex, field_strength,
                       manufacturer, icv):
        """
        Compute prediction + 95% prediction interval for one region.

        Parameters
        ----------
        region_label : str
            Must match the label in the Excel sheet, e.g. 'Hippocampus L'.
        age : float
            Age in years.
        sex : int
            1 = male, 0 = female.
        field_strength : int
            1 = 1.5T, 0 = 3T (same coding as the workbook).
        manufacturer : int
            1 = GE, 2 = Philips, 3 = Siemens.
        icv : float
            Estimated intracranial volume (as in the Excel file).

        Returns
        -------
        dict with keys:
            - 'pred'  : predicted (normative) volume in mm^3
            - 'lower' : lower bound of 95% prediction interval
            - 'upper' : upper bound of 95% prediction interval
            - 'se'    : standard error on the model scale
        """
        if region_label not in self.region_models:
            raise ValueError(
                f"Unknown region {region_label!r}. "
                f"Valid regions: {sorted(self.region_models)}"
            )

        model = self.region_models[region_label]
        covs = self._build_covariates(age, sex, field_strength, manufacturer, icv)

        # Build design vector x in the correct order for this region
        x = []
        for name in model["beta_names"]:
            key = str(name).lower()
            if key not in covs:
                raise KeyError(f"Missing covariate {name} (key {key})")
            x.append(covs[key])

        beta = model["beta_vals"]
        M = model["M"]
        mse = model["mse"]
        t = model["t"]

        # Linear predictor on the model scale
        lin = sum(b * xi for b, xi in zip(beta, x))

        # Quadratic form x' M x
        quad = 0.0
        for i in range(len(x)):
            for j in range(len(x)):
                quad += x[i] * M[i][j] * x[j]

        # Standard error of predicted *individual* (includes +1 term)
        se = math.sqrt(mse * (1.0 + quad))

        if not model["log10_model"]:
            # Direct scale
            pred = lin
            lower = lin - t * se
            upper = lin + t * se
        else:
            # Models on log10 scale: Ylog = X β
            # Excel does: 10^(LOG10(C) ± t * se), where C = 10^(Xβ).
            pred = 10 ** lin
            lower = 10 ** (lin - t * se)
            upper = 10 ** (lin + t * se)

        return {
            "pred": pred,
            "lower": lower,
            "upper": upper,
            "se": se,
        }

    def predict_all(self, age, sex, field_strength, manufacturer, icv):
        """
        Compute predictions + 95% prediction intervals for ALL regions.

        Returns
        -------
        dict
            {
              'Hippocampus L': {'pred': ..., 'lower': ..., 'upper': ..., 'se': ...},
              'Hippocampus R': {...},
              ...
            }
        """
        out = {}
        for label in self.region_models.keys():
            out[label] = self.predict_region(
                label, age, sex, field_strength, manufacturer, icv
            )
        return out
