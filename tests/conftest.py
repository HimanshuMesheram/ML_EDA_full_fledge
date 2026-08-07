import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_dataframe():
    """Fixture providing a sample raw Loan Default dataframe for unit tests."""
    data = {
        'ID': [1001, 1002, 1003, 1004, 1005],
        'year': [2019, 2019, 2019, 2019, 2019],
        'loan_amount': [200000, 350000, 150000, 500000, 250000],
        'property_value': [300000, 450000, np.nan, 600000, 300000],
        'income': [5000, 8000, 3000, 12000, np.nan],
        'term': [360, 360, 180, 360, 360],
        'dtir1': [35.0, 42.0, np.nan, 48.0, 25.0],
        'LTV': [66.67, 77.78, np.nan, 83.33, 83.33],
        'loan_type': ['type1', 'type2', 'type1', 'type3', 'type1'],
        'loan_purpose': ['p1', 'p3', 'p4', 'p1', 'p2'],
        'occupancy_type': ['pr', 'pr', 'ir', 'sr', 'pr'],
        'submission_of_application': ['to_inst', 'not_inst', 'to_inst', 'to_inst', 'not_inst'],
        'Neg_ammortization': ['not_neg', 'neg_amm', 'not_neg', 'not_neg', 'not_neg'],
        'lump_sum_payment': ['not_lpsm', 'not_lpsm', 'lpsm', 'not_lpsm', 'not_lpsm'],
        'co-applicant_credit_type': ['CIB', 'EXP', 'CIB', 'EXP', 'CIB'],
        'age': ['35-44', '45-54', '25-34', '55-64', '<25'],
        'total_units': ['1U', '1U', '2U', '1U', '1U'],
        'rate_of_interest': [3.5, 4.0, np.nan, 3.8, 4.2],
        'Interest_rate_spread': [0.5, 0.8, np.nan, 0.6, 0.9],
        'Upfront_charges': [1200, 2500, np.nan, 3000, 1500],
        'Credit_Score': [720, 680, 610, 750, 640],
        'construction_type': ['sb', 'sb', 'sb', 'sb', 'sb'],
        'Secured_by': ['home', 'home', 'home', 'home', 'home'],
        'Security_Type': ['direct', 'direct', 'direct', 'direct', 'direct'],
        'open_credit': ['nopc', 'nopc', 'nopc', 'nopc', 'nopc'],
        'Gender': ['Male', 'Female', 'Joint', 'Male', 'Female'],
        'Region': ['North', 'south', 'central', 'North', 'south'],
        'approv_in_adv': ['nopre', 'pre', 'nopre', 'nopre', 'nopre'],
        'loan_limit': ['cf', 'cf', 'ncf', 'cf', 'cf'],
        'business_or_commercial': ['nob/c', 'b/c', 'nob/c', 'nob/c', 'nob/c'],
        'credit_type': ['CIB', 'EXP', 'CRIF', 'EQUI', 'CIB'],
        'Credit_Worthiness': ['l1', 'l1', 'l2', 'l1', 'l1'],
        'interest_only': ['not_int', 'not_int', 'int_only', 'not_int', 'not_int'],
        'Status': [0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)
