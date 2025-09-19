import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime
import unicodedata
import re

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

    def _normalize_string(self, text: str) -> str:
        """
        Remove acentos e caracteres especiais de uma string.
        Mantém apenas letras, números e underscore.
        """
        # Remove acentos
        nfkd = unicodedata.normalize('NFKD', str(text))
        text_no_accents = ''.join([c for c in nfkd if not unicodedata.combining(c)])
        
        # Substitui caracteres especiais por underscore
        text_clean = re.sub(r'[^a-zA-Z0-9_]+', '_', text_no_accents)
        
        # Remove underscores duplicados e nas extremidades
        text_clean = re.sub(r'_+', '_', text_clean)
        text_clean = text_clean.strip('_')
        
        return text_clean.lower()

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
        Remove acentos e caracteres especiais.
        """
        parts = []
        
        # Normaliza o nome da variável
        parts.append(self._normalize_string(var))
        
        if category and category_value:
            parts.append(f"{self._normalize_string(category)}_{self._normalize_string(category_value)}")
            
        parts.append(agg.lower())
        
        if months:
            parts.append(f"{months}m")
            
        return "_".join(parts)

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
        additional_filters: str = None,
        ratio_vars: Dict[str, Union[str, List[str]]] = None,
        categorical_counts: Dict[str, List[str]] = None
    ) -> str:
        """
        Gera uma query otimizada usando uma única passada pela tabela.
        
        Args:
            numeric_vars: Lista de variáveis numéricas
            reference_date: Data de referência (formato YYYYMM)
            categorical_vars: Lista de variáveis categóricas para nunique e count
            time_breaks: Lista de períodos em meses para agregação
            date_format: Formato da data
            additional_filters: Filtros SQL adicionais
            ratio_vars: Dicionário {numerador: denominador(es)} para features de razão
            categorical_counts: Dicionário {coluna: [valores]} para contagens específicas
        """
        categorical_vars = categorical_vars or []
        time_breaks = time_breaks or [1, 3, 6, 12]
        ratio_vars = ratio_vars or {}
        categorical_counts = categorical_counts or {}
        
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

        # Features categóricas gerais
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

        # Features de contagens categóricas específicas
        for col, values in categorical_counts.items():
            col_normalized = self._normalize_string(col)
            for value in values:
                value_normalized = self._normalize_string(value)
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    time_filter = f"{self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    # Contagem para valor específico
                    safe_value = str(value).replace("'", "''")  # Escape single quotes
                    col_name = f"{col_normalized}_{value_normalized}_count_{months}m"
                    expression = f"SUM(CASE WHEN {time_filter} AND {col} = '{safe_value}' THEN 1 ELSE 0 END)"
                    features.append(f"{expression} AS {col_name}")
                    
                    # Percentual em relação ao total de transações
                    col_name_pct = f"{col_normalized}_{value_normalized}_pct_{months}m"
                    numerator = f"SUM(CASE WHEN {time_filter} AND {col} = '{safe_value}' THEN 1 ELSE 0 END)"
                    denominator = f"NULLIF(COUNT(CASE WHEN {time_filter} THEN 1 END), 0)"
                    features.append(f"100.0 * {numerator} / {denominator} AS {col_name_pct}")

        # Features de razão (ratio)
        for numerator_var, denominator_vars in ratio_vars.items():
            # Garante que denominator_vars seja sempre uma lista
            if isinstance(denominator_vars, str):
                denominator_vars = [denominator_vars]
            
            numerator_normalized = self._normalize_string(numerator_var)
            
            for denominator_var in denominator_vars:
                denominator_normalized = self._normalize_string(denominator_var)
                
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    col_name = f"ratio_{numerator_normalized}_{denominator_normalized}_{months}m"
                    numerator = f"SUM(CASE {time_filter} THEN {numerator_var} END)"
                    denominator = f"NULLIF(SUM(CASE {time_filter} THEN {denominator_var} END), 0)"
                    
                    features.append(f"{numerator} / {denominator} AS {col_name}")

        extra_where = f"\n    AND {additional_filters}" if additional_filters else ""
        
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

    def generate_categorical_count_features(
        self,
        categorical_counts: Dict[str, List[str]],
        reference_date: str,
        time_breaks: List[int] = None,
        date_format: str = '%Y%m',
        additional_filters: str = None
    ) -> str:
        """
        Método dedicado para gerar apenas features de contagem categóricas específicas.
        
        Args:
            categorical_counts: Dicionário onde chave é a coluna e valor é lista de categorias
                               Ex: {'tipo_transacao': ['DEBITO', 'CREDITO'], 'canal': ['ONLINE', 'LOJA']}
            reference_date: Data de referência (formato YYYYMM)
            time_breaks: Lista de períodos em meses para agregação
            date_format: Formato da data
            additional_filters: Filtros SQL adicionais
        
        Returns:
            Query SQL para gerar features de contagem categóricas
        """
        time_breaks = time_breaks or [1, 3, 6, 12]
        max_months = max(time_breaks)
        min_start_date = self._calculate_start_date(reference_date, max_months, date_format)
        
        features = []
        
        for col, values in categorical_counts.items():
            col_normalized = self._normalize_string(col)
            for value in values:
                value_normalized = self._normalize_string(value)
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    time_filter = f"{self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    # Escape single quotes in value
                    safe_value = str(value).replace("'", "''")
                    
                    # Contagem absoluta
                    col_name_count = f"{col_normalized}_{value_normalized}_count_{months}m"
                    count_expr = f"SUM(CASE WHEN {time_filter} AND {col} = '{safe_value}' THEN 1 ELSE 0 END)"
                    features.append(f"{count_expr} AS {col_name_count}")
                    
                    # Percentual
                    col_name_pct = f"{col_normalized}_{value_normalized}_pct_{months}m"
                    pct_expr = f"100.0 * SUM(CASE WHEN {time_filter} AND {col} = '{safe_value}' THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN {time_filter} THEN 1 END), 0)"
                    features.append(f"{pct_expr} AS {col_name_pct}")
                    
                    # Indicador binário (teve pelo menos uma ocorrência)
                    col_name_flag = f"{col_normalized}_{value_normalized}_flag_{months}m"
                    flag_expr = f"MAX(CASE WHEN {time_filter} AND {col} = '{safe_value}' THEN 1 ELSE 0 END)"
                    features.append(f"{flag_expr} AS {col_name_flag}")
        
        extra_where = f"\n    AND {additional_filters}" if additional_filters else ""
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

    def generate_ratio_features(
        self,
        ratio_vars: Dict[str, Union[str, List[str]]],
        reference_date: str,
        time_breaks: List[int] = None,
        date_format: str = '%Y%m'
    ) -> List[str]:
        """
        Gera features de razão entre variáveis usando dicionário.
        
        Args:
            ratio_vars: Dicionário onde chave é o numerador e valor é o(s) denominador(es)
                       Ex: {'valor_pago': 'valor_fatura', 'qtd_transacoes': ['qtd_dias', 'qtd_estabelecimentos']}
            reference_date: Data de referência
            time_breaks: Lista de períodos em meses
            date_format: Formato da data
        
        Returns:
            Lista de expressões SQL para features de razão
        """
        time_breaks = time_breaks or [1, 3, 6, 12]
        features = []
        
        for numerator_var, denominator_vars in ratio_vars.items():
            # Garante que denominator_vars seja sempre uma lista
            if isinstance(denominator_vars, str):
                denominator_vars = [denominator_vars]
            
            numerator_normalized = self._normalize_string(numerator_var)
            
            for denominator_var in denominator_vars:
                denominator_normalized = self._normalize_string(denominator_var)
                
                for months in time_breaks:
                    start_date = self._calculate_start_date(reference_date, months, date_format)
                    col_name = f"ratio_{numerator_normalized}_{denominator_normalized}_{months}m"
                    time_filter = f"WHEN {self.partition_column} >= '{start_date}' AND {self.partition_column} <= '{reference_date}'"
                    
                    numerator = f"SUM(CASE {time_filter} THEN {numerator_var} END)"
                    denominator = f"NULLIF(SUM(CASE {time_filter} THEN {denominator_var} END), 0)"
                    
                    features.append(f"{numerator} / {denominator} AS {col_name}")
        
        return features

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

    def generate_validation_query(self, reference_date: str = None) -> str:
        """
        Gera query para validação da qualidade dos dados.
        """
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
        """
        Retorna query para obter a última partição disponível.
        """
        return f"""
SELECT MAX({self.partition_column}) AS ultima_particao
FROM {self.base_name}
WHERE {self.partition_column} IS NOT NULL
"""