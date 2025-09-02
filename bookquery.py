import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime
#|import awswrangler as wr

class FeatureQueryGenerator:
    """
    Classe para gerar queries SQL de features agregadas por CPF para modelos de ML.
    Compatível com AWS Athena.
    """

    def __init__(
        self,
        base_name: str,
        partition_column: str = 'anomes',
        cpf_column: str = 'cpf',
        aggregations: List[str] = None
    ):
        """
        Inicializa o gerador de queries.
        
        Args:
            base_name: Nome da tabela/base no Athena
            partition_column: Nome da coluna de partição temporal (default: 'anomes')
            cpf_column: Nome da coluna de CPF (default: 'cpf')
            aggregations: Lista de agregações a serem usadas (default: ['sum', 'avg', 'max', 'min', 'count'])
        """
        self.base_name = base_name
        self.partition_column = partition_column
        self.cpf_column = cpf_column
        self.aggregations = aggregations or ['sum', 'avg', 'max', 'min', 'count']

    def _format_column_name(
        self, 
        var: str, 
        agg: str, 
        months: Optional[int] = None,
        category: Optional[str] = None,
        category_value: Optional[str] = None
    ) -> str:
        """
        Formata o nome da coluna de feature seguindo padrão consistente.
        """
        parts = [var]
        
        if category and category_value:
            parts.append(f"{category}_{category_value}")
            
        parts.append(agg)
        
        if months:
            parts.append(f"{months}m")
            
        return "_".join(parts).lower().replace(" ", "_").replace("-", "_")

    def _calculate_start_date(self, reference_date: str, months_back: int, date_format: str = '%Y%m') -> str:
        """
        Calcula a data de início baseada na data de referência e meses retroativos.
        """
        if date_format == '%Y%m':
            ref_year = int(reference_date[:4])
            ref_month = int(reference_date[4:6])
            
            start_month = ref_month - months_back + 1
            start_year = ref_year
            
            while start_month <= 0:
                start_month += 12
                start_year -= 1
            
            return f"{start_year:04d}{start_month:02d}"
        else:
            return reference_date

    def generate_optimized_query(
        self,
        numeric_vars: List[str],
        reference_date: str,
        categorical_vars: List[str] = None,
        time_breaks: List[int] = None,
        date_format: str = '%Y%m',
        additional_filters: str = None
    ) -> str:
        """
        Gera uma query otimizada usando uma única passada pela tabela.
        """
        categorical_vars = categorical_vars or []
        time_breaks = time_breaks or [1, 3, 6, 12]
        
        max_months = max(time_breaks)
        min_start_date = self._calculate_start_date(reference_date, max_months, date_format)
        
        features = []
        
        # Features numéricas
        for var in numeric_vars:
            for months in time_breaks:
                start_date = self._calculate_start_date(reference_date, months, date_format)
                time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                
                for agg in self.aggregations:
                    col_name = self._format_column_name(var, agg, months)
                    
                    if agg == 'sum':
                        expression = f"SUM(CASE {time_filter} THEN {var} ELSE 0 END)"
                    elif agg == 'std':
                         expression = f"STDDEV(CASE {time_filter} THEN {var} END)"
                    else:
                        expression = f"{agg.upper()}(CASE {time_filter} THEN {var} END)"
                    
                    features.append(f"{expression} AS {col_name}")

        # Features categóricas
        for var in categorical_vars:
            for months in time_breaks:
                start_date = self._calculate_start_date(reference_date, months, date_format)
                time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                
                # Nunique
                col_name_nunique = self._format_column_name(var, 'nunique', months)
                features.append(f"COUNT(DISTINCT CASE {time_filter} THEN {var} END) AS {col_name_nunique}")
                
                # Count
                col_name_count = self._format_column_name(var, 'count', months)
                features.append(f"COUNT(CASE {time_filter} THEN {var} END) AS {col_name_count}")

        extra_where = f"\n    AND {additional_filters}" if additional_filters else ""
        
        # **CORREÇÃO APLICADA AQUI**
        # Juntamos as features em uma string antes de passá-la para a f-string principal.
        features_sql = ',\n    '.join(features)

        query = f"""
SELECT 
    {self.cpf_column},
    '{reference_date}' AS dt_referencia,
    CURRENT_TIMESTAMP AS dt_processamento,
    {features_sql}
FROM {self.base_name}
WHERE {self.partition_column} >= '{min_start_date}'
    AND {self.partition_column} <= '{reference_date}'{extra_where}
GROUP BY {self.cpf_column}
"""
        return query

    # ... (o restante da classe não precisa de alterações, mas incluo por completude) ...
    def generate_query_with_categorical_breaks(
        self,
        numeric_vars: List[str],
        reference_date: str,
        categorical_breaks: List[str] = None,
        time_breaks: List[int] = None,
        date_format: str = '%Y%m',
        additional_filters: str = None
    ) -> str:
        """
        Gera query com quebras categóricas usando CTEs separadas.
        (Função não foi chamada no exemplo, mas corrigida para consistência)
        """
        categorical_breaks = categorical_breaks or []
        time_breaks = time_breaks or [1, 3, 6, 12]
        
        max_months = max(time_breaks)
        min_start_date = self._calculate_start_date(reference_date, max_months, date_format)
        
        ctes = []
        extra_where = f"\n    AND {additional_filters}" if additional_filters else ""

        # CTE base
        base_features = []
        for var in numeric_vars:
            for months in time_breaks:
                start_date = self._calculate_start_date(reference_date, months, date_format)
                time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                
                for agg in self.aggregations:
                    col_name = self._format_column_name(var, agg, months)
                    if agg == 'sum':
                        expression = f"SUM(CASE {time_filter} THEN {var} ELSE 0 END)"
                    else:
                        expression = f"{agg.upper()}(CASE {time_filter} THEN {var} END)"
                    base_features.append(f"{expression} AS {col_name}")

        # **CORREÇÃO APLICADA AQUI**
        base_features_sql = ',\n        '.join(base_features)
        base_cte = f"""base_features AS (
    SELECT 
        {self.cpf_column},
        {base_features_sql}
    FROM {self.base_name}
    WHERE {self.partition_column} >= '{min_start_date}'
        AND {self.partition_column} <= '{reference_date}'{extra_where}
    GROUP BY {self.cpf_column}
)"""
        ctes.append(base_cte)
        
        # CTEs para quebras categóricas
        for cat_break in categorical_breaks:
            cat_features = []
            for var in numeric_vars:
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    for agg in self.aggregations:
                        col_name = f"{cat_break}_{var}_{agg}_{months}m"
                        expression = f"{agg.upper()}(CASE {time_filter} THEN {var} END)"
                        cat_features.append(f"{expression} AS {col_name}")
            
            # **CORREÇÃO APLICADA AQUI**
            cat_features_sql = ',\n        '.join(cat_features)
            cat_cte = f"""features_{cat_break} AS (
    SELECT 
        {self.cpf_column},
        {cat_break},
        {cat_features_sql}
    FROM {self.base_name}
    WHERE {self.partition_column} >= '{min_start_date}'
        AND {self.partition_column} <= '{reference_date}'{extra_where}
    GROUP BY {self.cpf_column}, {cat_break}
)"""
            ctes.append(cat_cte)
        
        with_clause = "WITH " + ",\n".join(ctes)
        
        select_columns = ["bf.*", f"'{reference_date}' AS dt_referencia", "CURRENT_TIMESTAMP AS dt_processamento"]
        from_clause = "\nFROM base_features bf"
        
        for cat_break in categorical_breaks:
            select_columns.append(f"fc_{cat_break}.*")
            from_clause += f"""
LEFT JOIN features_{cat_break} fc_{cat_break}
    ON bf.{self.cpf_column} = fc_{cat_break}.{self.cpf_column}"""

        select_clause = "SELECT \n    " + ",\n    ".join(select_columns)
        
        return with_clause + "\n" + select_clause + from_clause
        
    def generate_ratio_features(
        self,
        numerator_vars: List[str],
        denominator_vars: List[str],
        reference_date: str,
        time_breaks: List[int] = None,
        date_format: str = '%Y%m'
    ) -> List[str]:
        time_breaks = time_breaks or [1, 3, 6, 12]
        features = []
        
        for num_var in numerator_vars:
            for den_var in denominator_vars:
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    col_name = f"ratio_{num_var}_{den_var}_{months}m"
                    time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    numerator = f"SUM(CASE {time_filter} THEN {num_var} END)"
                    denominator = f"NULLIF(SUM(CASE {time_filter} THEN {den_var} END), 0)"
                    
                    features.append(f"{numerator} / {denominator} AS {col_name}")
        
        return features

    def generate_validation_query(self, reference_date: str = None) -> str:
        date_filter = f"WHERE {self.partition_column} = '{reference_date}'" if reference_date else ""
        
        query = f"""
SELECT 
    '{reference_date}' AS dt_referencia,
    COUNT(DISTINCT {self.cpf_column}) AS qtd_cpfs_unicos,
    COUNT(*) AS qtd_total_registros,
    COUNT(*) - COUNT(DISTINCT {self.cpf_column}) AS qtd_duplicados,
    MIN({self.partition_column}) AS primeira_particao,
    MAX({self.partition_column}) AS ultima_particao,
    COUNT(DISTINCT {self.partition_column}) AS qtd_particoes,
    CASE 
        WHEN COUNT(*) - COUNT(DISTINCT {self.cpf_column}) > 0 
        THEN 'ALERTA: Existem registros duplicados'
        ELSE 'OK: Sem duplicados'
    END AS status_duplicados
FROM {self.base_name}
{date_filter}
"""
        return query

    def get_latest_partition_query(self) -> str:
        return f"""
SELECT MAX({self.partition_column}) AS ultima_particao
FROM {self.base_name}
WHERE {self.partition_column} IS NOT NULL
"""

    def execute_query(
        self,
        query: str,
        database: str,
        s3_output: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Executa a query no Athena usando awswrangler.
        """
        return wr.athena.read_sql_query(
            sql=query,
            database=database,
            s3_output=s3_output,
            **kwargs
        )

# Exemplo de uso
if __name__ == "__main__":
    generator = FeatureQueryGenerator(
        base_name="tb_transacoes",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    numeric_vars = ['valor_transacao']
    categorical_vars = ['tipo_transacao']
    time_breaks = [1, 3]
    
    query = generator.generate_optimized_query(
        numeric_vars=numeric_vars,
        categorical_vars=categorical_vars,
        time_breaks=time_breaks,
        reference_date='202412'
    )
    
    print("--- Query Gerada ---")
    print(query)