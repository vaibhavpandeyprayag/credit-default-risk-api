from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

num_cols = ['duration_months', 'credit_amount', 'age', 'credit_per_month']

ordinal_cols = ['checking_account_status', 'credit_history', 'savings_account'
]

nominal_cols = ['purpose', 'property', 'housing', 'employment_since',
# 'personal_status_sex', 'other_debtors', 'other_installment_plans'
]

ordinal_mapping = [
    ['A11','A14','A12','A13'],
    ['A31','A30','A32','A33','A34'],
    ['A65','A61','A62','A63','A64']
]

preprocessor_with_scaling = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("ord", OrdinalEncoder(categories=ordinal_mapping), ordinal_cols),
    ("nom", OneHotEncoder(drop="first"), nominal_cols)
])

preprocessor_without_scaling = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("ord", OrdinalEncoder(categories=ordinal_mapping), ordinal_cols),
    ("nom", OneHotEncoder(drop=None), nominal_cols)
])