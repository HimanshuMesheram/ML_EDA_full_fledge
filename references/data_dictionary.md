# Data Dictionary: Loan Default Dataset

## Raw Attributes (`data/raw/Loan_Default.csv`)

| Feature Name | Data Type | Category | Description | Decision |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | Integer | Identifier | Application unique key | **REMOVE** (Identifier key) |
| `year` | Integer | Temporal | Application tax year | **REMOVE** (Zero variance = 2019) |
| `loan_limit` | String | Categorical | Loan conforming limit (`cf` = conforming, `ncf` = non-conforming) | **REMOVE** (Low IV) |
| `Gender` | String | Categorical | Applicant gender (`Male`, `Female`, `Joint`, `Sex Not Available`) | **REMOVE** (Unpredictive) |
| `approv_in_adv` | String | Categorical | Pre-approval indicator (`pre`, `nopre`) | **REMOVE** (Low IV) |
| `loan_type` | String | Categorical | Mortgage loan type (`type1`, `type2`, `type3`) | **KEEP** (One-Hot Encode) |
| `loan_purpose` | String | Categorical | Purpose of loan (`p1`, `p2`, `p3`, `p4`) | **KEEP** (One-Hot Encode) |
| `Credit_Worthiness` | String | Categorical | Credit score tier (`l1`, `l2`) | **REMOVE** (Unpredictive) |
| `open_credit` | String | Categorical | Open credit status (`opc`, `nopc`) | **REMOVE** (99.6% near-zero variance) |
| `business_or_commercial` | String | Categorical | Commercial loan indicator (`b/c`, `nob/c`) | **REMOVE** (Low IV) |
| `loan_amount` | Float | Numerical | Requested principal loan amount ($) | **KEEP & TRANSFORM** (Log transform) |
| `rate_of_interest` | Float | Numerical | Interest rate (%) | **REMOVE** (Post-decision target leakage) |
| `Interest_rate_spread` | Float | Numerical | Interest rate spread over benchmark (%) | **REMOVE** (Post-decision target leakage) |
| `Upfront_charges` | Float | Numerical | Upfront origination fees ($) | **REMOVE** (Post-decision target leakage) |
| `term` | Float | Numerical | Repayment term in months (e.g. 360, 180) | **REMOVE** (Unpredictive) |
| `Neg_ammortization` | String | Categorical | Negative amortization clause (`neg_amm`, `not_neg`) | **KEEP** (One-Hot Encode) |
| `interest_only` | String | Categorical | Interest-only payment clause (`int_only`, `not_int`) | **REMOVE** (Low IV) |
| `lump_sum_payment` | String | Categorical | Lump sum payment clause (`lpsm`, `not_lpsm`) | **KEEP** (One-Hot Encode) |
| `property_value` | Float | Numerical | Appraised property value ($) | **KEEP & TRANSFORM** (Log transform & Impute) |
| `construction_type` | String | Categorical | Construction type (`sb`, `mh`) | **REMOVE** (99.98% near-zero variance) |
| `occupancy_type` | String | Categorical | Occupancy type (`pr` = principal, `sr`, `ir`) | **KEEP** (One-Hot Encode) |
| `Secured_by` | String | Categorical | Collateral backing type (`home`, `land`) | **REMOVE** (99.98% near-zero variance) |
| `total_units` | String | Categorical | Property units (`1U`, `2U`, `3U`, `4U`) | **KEEP & ENCODE** (Ordinal Encode) |
| `income` | Float | Numerical | Borrower gross monthly income ($) | **KEEP & TRANSFORM** (Log transform & Scale) |
| `credit_type` | String | Categorical | Credit bureau provider (`CIB`, `CRIF`, `EXP`, `EQUI`) | **REMOVE** (Low IV / Bureau artifact) |
| `Credit_Score` | Integer | Numerical | Credit score (500–900) | **REMOVE** (Unpredictive in dataset) |
| `co-applicant_credit_type` | String | Categorical | Co-applicant credit bureau (`CIB`, `EXP`) | **KEEP** (One-Hot Encode) |
| `age` | String | Categorical | Applicant age bracket (`<25`, `25-34`, ..., `>74`) | **KEEP & ENCODE** (Ordinal Encode) |
| `submission_of_application` | String | Categorical | Channel submission (`to_inst`, `not_inst`) | **KEEP** (One-Hot Encode) |
| `LTV` | Float | Numerical | Loan-to-Value ratio (%) | **KEEP** (Robust scale) |
| `Region` | String | Categorical | Geographical region (`North`, `south`, `central`, `North-East`) | **REMOVE** (Low IV) |
| `Security_Type` | String | Categorical | Security classification (`direct`, `Indriect`) | **REMOVE** (99.98% near-zero variance) |
| `Status` | Integer | Target | Binary target: 0 = Approved/Paid, 1 = Default/Rejected | **TARGET** |
| `dtir1` | Float | Numerical | Debt-to-Income ratio (%) | **KEEP & TRANSFORM** (Median impute) |

---

## Engineered Attributes (`src/features.py`)

| Feature Name | Data Type | Category | Formula / Construction | Business Justification |
| :--- | :--- | :--- | :--- | :--- |
| `property_value_isna` | Binary | Engineered | `1 if property_value.isnull() else 0` | Missing property appraisal flag (+0.41 correlation) |
| `dtir1_isna` | Binary | Engineered | `1 if dtir1.isnull() else 0` | Missing DTI ratio flag (+0.32 correlation) |
| `income_isna` | Binary | Engineered | `1 if income.isnull() else 0` | Missing income documentation flag |
| `LTV_isna` | Binary | Engineered | `1 if LTV.isnull() else 0` | Missing LTV ratio flag |
| `LTV_calculated` | Float | Engineered | `(loan_amount / property_value) * 100` | Recalculated clean LTV ratio excluding entry errors |
| `Payment_to_Income` | Float | Engineered | `(loan_amount / term) / (income / 12)` | Monthly debt burden relative to monthly income |
| `Loan_to_Income` | Float | Engineered | `loan_amount / income` | Leverage ratio relative to earning capacity |
| `DTI_x_LTV` | Float | Engineered | `dtir1 * LTV` | Compound risk surface interaction |
